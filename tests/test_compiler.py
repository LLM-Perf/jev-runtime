import pytest

from jev_runtime.schema import Option, Question


@pytest.mark.parametrize("count", [32, 64])
def test_large_joint_readout_handles_digit_splitting_tokenizers(compiler, bundle, count):
    question = Question(
        id="large",
        type="choice",
        instruction="Select the matching category.",
        options=tuple(Option(id=f"category{i}", description=f"Category {i}") for i in range(count)),
    )
    compiled = compiler.compile("A refund", question, bundle, "request")
    assert len(compiled.sequences) == 1
    sequence = compiled.sequences[0]
    assert len(sequence.label_ids) == len(set(sequence.label_ids)) == count
    assert compiled.keys == tuple(option.id for option in question.options)
