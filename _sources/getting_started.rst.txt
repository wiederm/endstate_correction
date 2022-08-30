Getting Started
===============
This page details how to get started with endstate_correction. 

Installation
-----------------
We recommend setting up a new python conda environment with :code:`python=3.9` and installing the packages defined `here <https://github.com/wiederm/endstate_correction/blob/main/devtools/conda-envs/test_env.yaml>`_ using :code:`mamba`.
This package can be installed using:
:code:`pip install git+https://github.com/wiederm/endstate_correction.git`.


How to use this package
-----------------
We have prepared two scripts that should help to use this package, both are located in :code:`endstate_correction/scripts`.
We will start by describing the use of the  :code:`sampling.py` script and then discuss the :code:`perform_correction.py` script.

A typical NEQ workflow
-----------------
Generate the equilibrium distribution :math:`\pi(x, \lambda=0)`
~~~~~~~~~~~~~~~~~~~~~~

In order to perform a NEQ work protocol, we need samples drawn from the equilibrium distribution from which we initialize our annealing moves.
If samples are not already available, the :code:`sampling.py` script provides and easy way to obtain these samples.

In the following we will use 1-octanol in a waterbox as a test system. Parameters, topology and initial coordinate set come with the :code:`endstate_correction` repository.
Subsequently, the relevant section of the :code:`sampling.py` script are explained --- but they should work for 1-octanol without any modifications. 

The scripts starts by defining an openMM system object. Here, CHARMM parameter, topology and coordinate files are used, but any other supported parameter set and files can be used. 
We start by defining a ``CharmmPsfFile``, ``PDBFile`` and ``CharmmParameterSet``:  

.. code::

    psf = CharmmPsfFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.psf")
    pdb = PDBFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.pdb")
    params = CharmmParameterSet(
        f"{parameter_base}/{system_name}/charmm-gui/unk/unk.rtf",
        f"{parameter_base}/{system_name}/charmm-gui/unk/unk.prm",
        f"{parameter_base}/toppar/top_all36_cgenff.rtf",
        f"{parameter_base}/toppar/par_all36_cgenff.prm",
        f"{parameter_base}/toppar/toppar_water_ions.str",
    )

and then we define the atoms that should be perturbed using the coupling parameter :math:`\lambda` with

.. code:: python

    ml_atoms = [atom.index for atom in chains[0].atoms()]

Depending if all atoms in your system are included in the :code:`ml_atoms` list or only a subset, you can set up your QML or QML/MM simulation object using 

.. code:: python

    sim = create_charmm_system(psf=psf, parameters=params, env=env, ml_atoms=ml_atoms)


That is everything you need to define to run the equilibrium sampling. 
The parameters controling the number of samples to save and time interval between samples can be set in the script in the relevant parts.
Keep in mind that if you want to perform bidirectional FEP or NEQ you need to sample at :math:`\pi(x, \lambda=0)` *and* :math:`\pi(x, \lambda=1)`. 
This can be controlled by setting the number using the variable :code:`nr_lambda_states`.
The default value set in the :code:`sampling.py` script is :code:`nr_lambda_states=2`, generating samples from both endstate equilibrium distributions.

Perform unidirectional NEQ from :math:`\pi(x, \lambda=0)`
~~~~~~~~~~~~~~~~~~~~~~
The endstate correction can be performed using the script :code:`perform_correction.py`.
1-octanol in a waterbox will be the test system again. Parameters, topology and initial coordinate set come with the :code:`endstate_correction` repository.
Subsequently, the relevant section of the :code:`perform_correction.py` script are explained --- but they should work for 1-octanol without any modifications. 

To perform a specific endstate correction we need to define a protocol 
(some standard protocols are shown :ref:`here<Available protocols>`) 
with:

.. code:: python

  neq_protocol = Protocol(
      method="NEQ",
      direction="unidirectional",
      sim=sim,
      trajectories=[mm_samples],
      nr_of_switches=400,
      neq_switching_length=5_000, # in fs
  )

This protocol is then passed to the actual function performing the protocol: :code:`perform_endstate_correction(neq_protocol)`.

Perform bidirectional NEQ from :math:`\pi(x, \lambda=0)` and :math:`\pi(x, \lambda=1)`
~~~~~~~~~~~~~~~~~~~~~~
The endstate correction can be performed using the script :code:`perform_correction.py` and the following protocol.

.. code:: python

  neq_protocol = Protocol(
      method="NEQ",
      direction="bidirectional",
      sim=sim,
      trajectories=[mm_samples, qml_samples],
      nr_of_switches=400,
      neq_switching_length=5_000, # in fs
  )

This protocol is then passed to the actual function performing the protocol: :code:`perform_endstate_correction(neq_protocol)`.


Perform unidirectional FEP from :math:`\pi(x, \lambda=0)`
~~~~~~~~~~~~~~~~~~~~~~
The endstate correction can be performed using the script :code:`perform_correction.py`.
The protocol has to be adopted slightly:

.. code:: python

  fep_protocol = Protocol(
      method="FE{",
      direction="unidirectional",
      sim=sim,
      trajectories=[mm_samples],
      nr_of_switches=400,
  )
This protocol is then passed to the actual function performing the protocol: :code:`perform_endstate_correction(fep_protocol)`.


Analyse results of an unidirection NEQ protocol
~~~~~~~~~~~~~~~~~~~~~~
To analyse the results generated by :code:`r = perform_endstate_correction()` pass the return value to :code:`plot_endstate_correction_results(system_name, r, "results_neq_unidirectional.png")` and results will be plotted and printed.


Available protocols
-----------------

.. code:: python

  fep_protocol = Protocol(
      method="FEP",
      direction="unidirectional",
      sim=sim,
      trajectories=[mm_samples],
      nr_of_switches=400,
  )

.. code:: python

  fep_protocol = Protocol(
      method="FEP",
      direction="bidirectional",
      sim=sim,
      trajectories=[mm_samples, qml_samples],
      nr_of_switches=400,
  )

.. code:: python

  neq_protocol = Protocol(
      method="NEQ",
      direction="unidirectional",
      sim=sim,
      trajectories=[mm_samples],
      nr_of_switches=400,
      neq_switching_length=5_000, # in fs
  )

.. code:: python

  neq_protocol = Protocol(
      method="NEQ",
      direction="bidirectional",
      sim=sim,
      trajectories=[mm_samples, qml_samples],
      nr_of_switches=400,
      neq_switching_length=5_000, # in fs
  )
