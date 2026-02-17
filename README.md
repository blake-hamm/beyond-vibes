# Beyond Vibes

This project provides a framework to evaluate local models and compare them in latency and quality across various tasks. Of course, this can also be used to test model api vendors, but the focus is local models with unique requirements.


## Use cases

The primary use cases for local models are agentic coding and chat/search capabilities. We will setup tests in [OpenCode](https://opencode.ai/docs/) that evaluate:

- Architecture and planning
- Unit test creation
- New feature creation
- Vendor research and searching

## Methodology

In order to test these use cases, we plan a 'scientific' methodology. First off, we setup our experiment by targeting specific repos and a task for each use case. We use OpenCode and provide instructions with a desired outcome in mind. This establishes a 'golden dataset' with pre-defined inputs and desired outputs.

To form the golden outputs, we manually walk through the tasks (inputs) with a human-in-the-loop and a more powerful model like Opus 4.6. Once we are satisfied with the desired outcome of our tasks after some hand-holding, we ensure the same tasks can be executed autonomously using OpenCode.

Now that we have our dataset, we will need a way to evaluate different outcomes. To do this, we use DSPy and create an LLM judge. We can use the golden dataset to optimize the prompt for this judge to ensure it correctly judges our outcomes. The final score is a collection of smaller judges that creates the following datapoints:

- Compare
  - Does the output or code execute the same logic?
  - Were there any aspects of the solution or steps along the way that totally missed?
  - If logic is correct, how "idiomatic" or "clean" is it compared to the Golden solution
- Tools
  - Deterministic - percent of successful tool calls?
  - Eventually, were tool calls completed successfully?
- Vibe
  - On a scale of 1-5, how sycophantic was the response?
  - Was there some extreme hallucinations made that aren't relevant or factually true at all?
  - Was the response direct and solved the problem swiftly without extra steps?
