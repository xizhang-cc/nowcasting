
================
Introduction
================
Precipitation nowcasting using IMERG data in West Africa



================
Pipeline
================
To share an existing environment.yml file:
1. Activate the environment to export
.. code-block:: bash
    $ conda activate myenv
2. Export your active environment to a new file
.. code-block:: bash
    $ conda env export > environment.yml


To create a new environment from an environment.yml file:
1. Create the environment from the environment.yml file
.. code-block:: bash
    $ conda env create -f environment.yml
Note that the first line of the yml file sets the new environment's name, feel free to change if you want.
2. Activate the new environment
.. code-block:: bash
    $ conda activate myenv



