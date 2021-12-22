# Iguanas docker container (needs Zscaler disabled)
# If getting ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE 
# REQUIREMENTS FILE, just keep re-running

# Install default environment
FROM dockerhub.paypalcorp.com/simility/python:3.7

# Delete cache (stops ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE 
# REQUIREMENTS FILE)
RUN rm ~/.cache/pip -rf

# Install Java
RUN apt update -y && apt-get install -y software-properties-common && \
    apt-add-repository 'deb http://security.debian.org/debian-security stretch/updates main' && apt update -y && \
    apt-get install -y openjdk-8-jdk-headless && \    
    export JAVA_HOME && \
    apt-get clean

# Install python libs
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y python python-dev python-pip virtualenv libssl-dev libpq-dev git build-essential libfontconfig1 libfontconfig1-dev

# Copy Iguanas directory
RUN mkdir /iguanas
COPY ./ /iguanas

# Set certificates
RUN pip config set global.cert /iguanas/combined_cacerts.pem

# Upgrade setuptools and pip
RUN pip install setuptools pip --upgrade --force-reinstall --no-cache-dir

# Set up working directory
WORKDIR /
RUN mkdir /workdir/
RUN mkdir /workdir/examples

# Copy over examples from iguanas folder 
COPY ./iguanas/correlation_reduction/examples /workdir/examples/subpackage_examples/correlation_reduction
COPY ./iguanas/rbs/examples /workdir/examples/subpackage_examples/rbs
COPY ./iguanas/rule_application/examples /workdir/examples/subpackage_examples/rule_application
COPY ./iguanas/rule_generation/examples /workdir/examples/subpackage_examples/rule_generation
COPY ./iguanas/rule_optimisation/examples /workdir/examples/subpackage_examples/rule_optimisation
COPY ./iguanas/rule_scoring/examples /workdir/examples/subpackage_examples/rule_scoring
COPY ./iguanas/rule_selection/examples /workdir/examples/subpackage_examples/rule_selection
COPY ./iguanas/rules/examples /workdir/examples/subpackage_examples/rules

# Install sub-packages
RUN pip install /iguanas/iguanas/correlation_reduction/. --no-cache-dir
RUN pip install /iguanas/iguanas/rbs/. --no-cache-dir
RUN pip install /iguanas/iguanas/rule_application/. --no-cache-dir
RUN pip install /iguanas/iguanas/rule_generation/. --no-cache-dir
RUN pip install /iguanas/iguanas/rule_optimisation/. --no-cache-dir
RUN pip install /iguanas/iguanas/rule_scoring/. --no-cache-dir
RUN pip install /iguanas/iguanas/rule_selection/. --no-cache-dir
RUN pip install /iguanas/iguanas/rules/. --no-cache-dir
RUN pip install /iguanas/iguanas/utils/. --no-cache-dir
# Install Iguanas
RUN pip install /iguanas/. --no-cache-dir

# Install jupyter packages
RUN pip install jupyterlab --no-cache-dir
RUN pip install ipykernel --no-cache-dir

ENTRYPOINT jupyter lab --ip=0.0.0.0 --port=8080 --allow-root --no-browser --notebook-dir /workdir/ --NotebookApp.token=''
