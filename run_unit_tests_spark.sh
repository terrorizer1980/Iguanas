# find . -name "test_*.py" -name "*spark*" -exec python -m pytest "{}" \;
pytest ./iguanas/*/tests/*spark* 