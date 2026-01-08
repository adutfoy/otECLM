oteclm module
=============

oteclm is an OpenTURNS module. It allows the treatment of failure probabilities and dependencies in highly redundant systems according to the Extended Common Load Model (ECLM) .


Install from git
================

Get the archive:

.. code-block:: shell

    $ pip install git+https://github.com/adutfoy/otECLM.git
 

Install from source
===================

Get source:

.. code-block:: shell

    $ git clone https://www.github.com/adutfoy/otECLM.git


The install procedure is performed as follows:

.. code-block:: shell

    $ pip install .

To run the tests:

.. code-block:: shell

    $ pytest

Finally to build the documentation, you should install the documentation requirements :

.. code-block:: shell

    $ pip install .[doc]

This builds the documentation in the `build` folder. Another option is to launch the `make` command:

.. code-block:: shell

    $ make html -C doc

Documentation
=============

A documentation, including examples is available `here <https://adutfoy.github.io/sphinx/oteclm/main/>`_.
