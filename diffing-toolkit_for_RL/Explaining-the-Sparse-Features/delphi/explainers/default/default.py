import asyncio
from dataclasses import dataclass

from ..explainer import ActivatingExample, Explainer
from .prompt_builder import build_prompt


@dataclass
class DefaultExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""
    cot: bool = False
    """Whether to use chain of thought reasoning."""
    sentence_level: bool = False
    """Whether to highlight at the sentence level."""

    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        highlighted_examples = []

        for i, example in enumerate(examples):
            str_toks = example.str_tokens
            activations = example.activations.tolist()
            highlighted_examples.append(f'Example {i+1}: ' + self._highlight(str_toks, activations, self.sentence_level).replace("\n", "\\n"))

            # import pdb; pdb.set_trace()

            if self.activations:
                assert (
                    example.normalized_activations is not None
                ), "Normalized activations are required for activations in explainer"
                normalized_activations = example.normalized_activations.tolist()
                highlighted_examples.append(
                    self._join_activations(
                        str_toks, activations, normalized_activations, self.sentence_level
                    )
                )

        highlighted_examples = "\n".join(highlighted_examples)

        prompt = build_prompt(
            examples=highlighted_examples,
            activations=self.activations,
            cot=self.cot,
            sentence_level=self.sentence_level,
        )

        # import pdb; pdb.set_trace()

        # print(prompt, flush=True)
        return prompt

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
