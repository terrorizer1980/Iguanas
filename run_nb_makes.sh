# find iguanas/*/examples -name "*example.ipynb" ! -name "*spark*" -exec pytest "{}" --nbmake --nbmake-kernel=iguanas_no_spark \;
shopt -s extglob
pytest ./examples/!(*spark*) --nbmake --nbmake-kernel=iguanas_no_spark
pytest ./iguanas/*/examples/!(*spark*) --nbmake --nbmake-kernel=iguanas_no_spark
shopt -u extglob