import argparse
import os

import torch
from tqdm import tqdm

EMBEDDING_PATH = "embeddings"

lang_dict = {
    "codebert": ["python", "java", "javascript", "php", "ruby", "go"],
    "codellama": ["python", "java", "javascript", "php", "c++", "c-sharp"],
    "codet5": [
        "python",
        "java",
        "javascript",
        "php",
        "ruby",
        "c",
        "c++",
        "c-sharp",
        "go",
    ][:4],
}


def T(mat):
    return torch.transpose(mat, 0, 1)


def embeddings_parser():
    parser = argparse.ArgumentParser(
        description="Generate Syntax specific components / matrices."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=["codebert", "codet5", "codellama"],
        help="The model type that sourced the embeddings: codebert, codet5, or codellama",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--sourceDevice",
        type=str,
        required=False,
        choices=["cpu", "cuda"],
        help="Rank of Low Rank Decomposition / Common Specific Low Rank Decomposition",
    )
    parser.add_argument(
        "--doCentering",
        action="store_true",
        help="Whether to generate the Centering syntax specific component.",
    )
    parser.add_argument(
        "--doLRD",
        action="store_true",
        help="Whether to generate the Low Rank Decomposition syntax matrices ",
    )
    parser.add_argument(
        "--doCSLRD",
        action="store_true",
        help="Whether to generate the Common Specific Low Rank Decomposition syntax matrix.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank for Low Rank Decomposition / Common Specific Low Rank Decomposition",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=1000,
        help="Number of embeddings to use to calculate syntax specific components",
    )
    return parser


if __name__ == "__main__":
    parser = embeddings_parser()

    args = parser.parse_args()

    languages = lang_dict[args.model]
    model_type = args.model
    model_path = os.path.join(EMBEDDING_PATH, model_type)

    device = args.device
    sDevice = args.sourceDevice
    r = args.rank

    if (args.rank is None or args.rank == 0) and (args.doCSLRD or args.doLRD):
        parser.error("--doLRD and --doCSLRD require a nonzero --rank")
    if sDevice is None:
        sDevice = device
    if args.doCSLRD and args.rank > len(languages):
        parser.error(
            "--rank for CSLRD cannot be greater than model's known languages ("
            + str(len(languages))
            + ")"
        )

    if args.doCentering or args.doLRD:
        for lang in languages:
            embed_path = os.path.join(model_path, lang, lang)

            tensors = [
                torch.load(
                    f"{embed_path}{i}.txt.pt",
                    weights_only=True,
                    map_location=torch.device(sDevice),
                )
                for i in tqdm(range(args.number), desc=f"Loading {lang} snippets")
            ]
            if model_type == "codellama":
                # codellama embeddings have not yet been mean-pooled across tokens
                # mean-pool each tensor along the first dimension (n, 4096 -> 1, 4096)
                tensors = [torch.mean(tensor, dim=0) for tensor in tensors]

            if args.doCentering:
                mean_embedding = torch.unsqueeze(torch.mean(torch.stack(tensors), 0), 0)
                torch.save(
                    mean_embedding,
                    os.path.join(model_path, model_type + "_mean_" + lang + ".pt"),
                )

            if args.doLRD:
                _, _, Vh = torch.linalg.svd(torch.stack(tensors))
                P = torch.matmul(Vh[:, :r], T(Vh[:, :r]))
                torch.save(
                    P,
                    os.path.join(
                        model_path, model_type + "_LRD" + str(r) + "_" + lang + ".pt"
                    ),
                )

    if args.doCSLRD:
        means = []
        l = len(languages)

        for lang in languages:
            file_path = os.path.join(model_path, model_type + "_mean_" + lang + ".pt")
            try:
                mean_embedding = torch.load(
                    file_path, weights_only=True, map_location=torch.device(sDevice)
                )[0]
            except FileNotFoundError:
                print(
                    "FileNotFoundError: No mean language embedding found at "
                    + file_path
                    + ". Run with doCentering as True to generate these files."
                )
                exit()
            means.append(mean_embedding)

        M = T(torch.stack(means))
        d = len(M)

        mc = torch.matmul(M / d, torch.ones(l)).reshape(-1, 1)

        Ur, Sr, Vr = torch.linalg.svd(M - torch.matmul(mc, torch.ones((1, l))))

        Ms = Ur[:, :r]
        gamma = torch.matmul(T(Vr)[:, :r], torch.diag(Sr[:r]))

        Ms = torch.matmul(mc, torch.ones((1, l))) + torch.matmul(Ms, T(gamma))

        mc = torch.matmul(T(torch.linalg.pinv(Ms)), torch.ones(l))
        mc = mc / torch.sum(mc**2)

        prod = Ms - torch.matmul(mc.reshape(-1, 1), torch.ones((1, l)))
        U, _, _ = torch.linalg.svd(prod)
        MsMsT = torch.matmul(U[:, :r], T(U[:, :r]))
        torch.save(
            MsMsT, os.path.join(model_path, model_type + "_CSLRD" + str(r) + ".pt")
        )
