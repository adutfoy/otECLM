oteclm module
=============

oteclm is an OpenTURNS module. It allows the treatment of failure probabilities and dependencies in highly redundant systems according to the Extended Common Load Model (ECLM) .


Build from archive
==================

Get the archive:

.. code-block:: shell

    $ wget https://github.com/adutfoy/otECLM/archive/main.zip -O otECLM-main.zip
 

The install procedure is performed as follows:

.. code-block:: shell

    $ python install otECLM-main.zip
   

Build from source
=================

Get source:

.. code-block:: shell

    $ git clone https://www.github.com/adutfoy/otECLM.git


The install procedure is performed as follows:

.. code-block:: shell

    $ python setup.py install

If you need to install the module in the user folder :

.. code-block:: shell

    $ python setup.py install --user

To run the tests:

.. code-block:: shell

    $ pytest

Finally to build the documentation, you should invoke the `build_sphinx` option :

.. code-block:: shell

    $ python setup.py build_sphinx

This builds the documentation in the `build` folder. Another option is to launch the `make` command:

.. code-block:: shell

    $ make html -C doc

Documentation
=============

A documentation, including examples is available `here <https://adutfoy.github.io/sphinx/oteclm/main/>`_.
