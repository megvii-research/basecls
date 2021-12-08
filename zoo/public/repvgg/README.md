# RepVGG Series

Use `RepVGG.convert_to_deploy(model)` to convert a training RepVGG to deploy:

```python
model = RepVGG(..., deploy=False)
model.load_state_dict(...)
_ = RepVGG.convert_to_deploy(model)
```
