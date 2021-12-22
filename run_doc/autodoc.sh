cd ..
rm -rf _docs
rm -rf docs
mkdir _docs
mkdir docs
sphinx-apidoc -F -M -d 1 --separate -o _docs iguanas
cd _docs
rm iguanas*rst

# Copy over website structure
cp -r ../run_doc/files/* .
# Copy over example notebooks
cp ../examples/*example.ipynb ./examples/complete
cp ../iguanas/correlation_reduction/examples/*example.ipynb ./examples/correlation_reduction
cp ../iguanas/metrics/examples/*example.ipynb ./examples/metrics
cp ../iguanas/pipeline/examples/*example.ipynb ./examples/pipeline
cp ../iguanas/rbs/examples/*example.ipynb ./examples/rbs
cp ../iguanas/rule_application/examples/*example.ipynb ./examples/rule_application
cp ../iguanas/rule_generation/examples/*example.ipynb ./examples/rule_generation
cp ../iguanas/rule_optimisation/examples/*example.ipynb ./examples/rule_optimisation
cp ../iguanas/rule_scoring/examples/*example.ipynb ./examples/rule_scoring
cp ../iguanas/rule_selection/examples/*example.ipynb ./examples/rule_selection
cp ../iguanas/rules/examples/*example.ipynb ./examples/rules
make clean
make html
cd ..
touch _docs/_build/html/.nojekyll
cp -r _docs/_build/html/* docs
rm -r _docs