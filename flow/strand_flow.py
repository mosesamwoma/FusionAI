from strand import Node, Flow
from model.client import generate
from config.settings import MODELS, FUSION_MODEL


def build_flow(prompt):

    # Create model nodes dynamically
    model_nodes = []

    for m in MODELS:

        def model_fn(p=prompt, provider=m["provider"], model=m["model"]):
            return generate(provider, model, p)

        model_nodes.append(Node(model_fn))

    # Fusion node
    def fusion_fn(*responses):
        valid = [r for r in responses if "Error:" not in r]

        combined = "\n\n".join(
            [f"Answer {i+1}:\n{r}" for i, r in enumerate(valid)]
        )

        fusion_prompt = f"""
You are an AI judge.

Combine the following answers into one improved response.

Question:
{prompt}

{combined}
"""

        return generate(
            FUSION_MODEL["provider"],
            FUSION_MODEL["model"],
            fusion_prompt
        )

    fusion_node = Node(fusion_fn)

    # Build flow
    flow = Flow()

    for node in model_nodes:
        flow.connect(node, fusion_node)

    return flow.run()
