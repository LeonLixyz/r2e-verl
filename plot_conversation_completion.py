#!/usr/bin/env python3

from verl.workers.reward_manager.rllm import RLLMRewardManager

# Create a dummy tokenizer (not used for analysis)
class DummyTokenizer:
    def decode(self, *args, **kwargs):
        return ""

# Create RLLM reward manager instance
reward_manager = RLLMRewardManager(
    tokenizer=DummyTokenizer(),
    num_examine=1
)

# Analyze and plot conversation completion
print("Analyzing conversation logs...")
results = reward_manager.analyze_and_plot_conversation_completion("./conversation_logs")

if results:
    print(f"\nAnalysis complete!")
    print(f"Finished responses: {results['finished']} ({results['finished_percentage']:.1f}%)")
    print(f"Truncated responses: {results['truncated']} ({results['truncated_percentage']:.1f}%)")
    print(f"Total responses: {results['total']}")
else:
    print("No conversation logs found or analysis failed.")
