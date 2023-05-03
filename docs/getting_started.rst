Getting Started
===============
This page details how to perform free energy calculations between a molecular mechanics force field (in this case the open-forcefield is sued) and a neural network potential (ANI-2x).

Installation
-----------------
We recommend setting up a new python conda environment with :code:`python=3.9` and installing the packages defined `here <https://github.com/wiederm/endstate_correction/blob/main/devtools/conda-envs/test_env.yaml>`_ using :code:`mamba`.
This package can be installed using:
:code:`pip install git+https://github.com/wiederm/endstate_correction.git`.


How to use this package
-----------------
We have prepared two scripts that should help to use this package, both are located in :code:`endstate_correction/scripts`.
We will start by describing the use of the :code:`sampling.py` script and then discuss the :code:`perform_correction.py` script.

A typical NEQ workflow
-----------------
Generate the equilibrium distribution :math:`\pi(x, \lambda=0)`
~~~~~~~~~~~~~~~~~~~~~~

In order to perform a NEQ work protocol, we need samples drawn from the equilibrium distribution from which we initialize our annealing simulations.
If samples are not already available, the :code:`generate_endstate_samples.py` script provides and easy way to obtain these samples.

In the following we will use a molecule from the HiPen dataset with the SMILES string ""Cn1cc(Cl)c(/C=N/O)n1" as a test system. 

The scripts starts by defining some control parameters such as ``n_samples`` (number of samples that should be generated), ``n_steps_per_sample`` (integration steps between samples), ``base`` (where to store the samples).

We then start setting the simulation object 

.. code:: python

    forcefield = ForceField("openff_unconstrained-2.0.0.offxml")
    env = "vacuum"

    potential = MLPotential("ani2x")
    molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False) # generate molecule from smile
    molecule.generate_conformers(n_conformers=1) # generate a single conforamtion

    topology = molecule.to_topology()
    system = forcefield.create_openmm_system(topology) 
    # define region that should be treated with the nnp
    ml_atoms = [atom.molecule_particle_index for atom in topology.atoms]
    # set up the integrator and the platform (here we are assuming you have a CUDA GPU)
    integrator = LangevinIntegrator(temperature, collision_rate, stepsize)
    platform = Platform.getPlatformByName("CUDA")
    # we are creating a mixed system using openmm-ml
    topology = topology.to_openmm()
    ml_system = potential.createMixedSystem(topology, system, ml_atoms, interpolate=True)
    sim = Simulation(topology, ml_system, integrator, platform=platform)

Note that we explicitly define the atoms that should be perturbed from the reference to the target potential using 

.. code:: python

    ml_atoms = [atom.molecule_particle_index for atom in topology.atoms]

If you want to perform bidirectional FEP or NEQ you need to sample at :math:`\pi(x, \lambda=0)` *and* :math:`\pi(x, \lambda=1)`. 
This can be controlled by setting the number using the variable :code:`nr_lambda_states`.
The default value set in the :code:`sampling.py` script is :code:`nr_lambda_states=2`, generating samples from both endstate equilibrium distributions.

.. code:: python
    nr_lambda_states = 2  # samples equilibrium distribution at both endstates
    lambs = np.linspace(0, 1, nr_lambda_states)

Perform unidirectional NEQ from :math:`\pi(x, \lambda=0)`
~~~~~~~~~~~~~~~~~~~~~~
The endstate correction can be performed using the script :code:`perform_correction.py`.
The script will calculate the free energy estimate using the samples generated with the :code:`generate_endstate_samples.py` script.
Subsequently, the relevant section of the :code:`perform_correction.py` script are explained --- but they should work for for the testsystem without any modifications. 

To perform a specific endstate correction we need to define a protocol 
(some standard protocols are shown :ref:`here<Available protocols>`) 
with:

.. code:: python

    neq_protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        nr_of_switches=100,
        neq_switching_length=1_000,
    )

This protocol is then passed to the actual function performing the protocol: :code:`perform_endstate_correction(neq_protocol)`.
The particular code above defines unidirectional NEQ switching using 100 switches and a switching length of 1 ps.
The direciton of the switching simulation is controlled by the sampels that are provided: 
if `reference_samples` are provided, switching is performed from the reference to the target level of theory, if `target_samples` are provided, switching is performed from the target level to the reference level.
If both samples are provided, bidirectional NEQ switching is performed (for an example see below).

Perform bidirectional NEQ from :math:`\pi(x, \lambda=0)` and :math:`\pi(x, \lambda=1)`
~~~~~~~~~~~~~~~~~~~~~~
The endstate correction can be performed using the script :code:`perform_correction.py` and the following protocol.

.. code:: python

    neq_protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=qml_samples,
        nr_of_switches=100,
        neq_switching_length=1_000,
    )

This protocol is then passed to the actual function performing the protocol: :code:`perform_endstate_correction(neq_protocol)`.


Perform unidirectional FEP from :math:`\pi(x, \lambda=0)`
~~~~~~~~~~~~~~~~~~~~~~
The protocol has to be adopted slightly:

.. code:: python
    fep_protocol = Protocol(
        method="FEP",
        sim=sim,
        reference_samples=mm_samples,
        nr_of_switches=1_000,
    )

This protocol is then passed to the actual function performing the protocol: :code:`perform_endstate_correction(fep_protocol)`.


Analyse results of an unidirection NEQ protocol
~~~~~~~~~~~~~~~~~~~~~~
To analyse the results generated by :code:`r = perform_endstate_correction()` pass the return value to :code:`plot_endstate_correction_results(system_name, r, "results_neq_unidirectional.png")` and results will be plotted and printed.


Available protocols
-----------------

.. code:: python

    neq_protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=qml_samples,
        nr_of_switches=100,
        neq_switching_length=1_000,
    )

.. code:: python

    neq_protocol = Protocol(
        method="FEP",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=qml_samples,
        nr_of_switches=100,
        neq_switching_length=1_000,
        save_endstates=True,
        save_trajs=True,
    )

