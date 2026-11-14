# 7. Generation: when the LLM has the answer and still gets it wrong

The last stage is the LLM writing the answer. Retrieval can be perfect and this can still fail.

**A real case (q021):** "A new Docker Desktop version came out and I'm on RHEL. Can I just install the new
RPM over my current one?" The correct passage ("You need to first remove the previous version...")
was found, ranked first at every stage, and packed first. The bot still opened with
**"Yes, you can install the new RPM over your current one,"** and only then described removing the old
version, contradicting itself.

ragblame's **oracle test** hands the model only the correct passage. Here the oracle answer was right,
so the model *can* read it. With the other passages present it got confused, which ragblame labels
**distracted**. When even the oracle answer is wrong, the cause is **generation**: the model can't use
the evidence, and retrieval can't fix that.

Suggested fix for both: a prompt that makes the model quote the supporting line first, then answer.
