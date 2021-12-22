# find . -name "test_*.py" ! -name "*spark*" -exec python -m pytest "{}" \;
shopt -s extglob
pytest ./iguanas/*/tests/!(*spark*)
shopt -u extglob