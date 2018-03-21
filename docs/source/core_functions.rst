Core Functions
=======================

Functions that are required to operate the package at a basic level

.. autosummary::

   caiman.source_extraction.cnmf.pre_processing.preprocess_data

   caiman.source_extraction.cnmf.initialization.initialize_components

   caiman.source_extraction.cnmf.spatial.update_spatial_components

   caiman.source_extraction.cnmf.temporal.update_temporal_components

   caiman.source_extraction.cnmf.merging.merge_components

   caiman.base.movies.movie

   caiman.utils.visualization.plot_contours

   caiman.utils.visualization.view_patches_bar

   caiman.source_extraction.cnmf.utilities.order_components

   caiman.source_extraction.cnmf.utilities.manually_refine_components


Preprocessing
---------------
.. currentmodule:: caiman.source_extraction.cnmf.pre_processing

.. autofunction:: preprocess_data


Initialization
---------------
.. currentmodule:: caiman.source_extraction.cnmf.initialization

.. autofunction:: initialize_components


Spatial Components
-------------------
.. currentmodule:: caiman.source_extraction.cnmf.spatial

.. autofunction:: update_spatial_components


Temporal Components
-------------------
.. currentmodule:: caiman.source_extraction.cnmf.temporal

.. autofunction:: update_temporal_components



Merge components
----------------
.. currentmodule:: caiman.source_extraction.cnmf.merging

.. autofunction:: merge_components

Visualization
-------------
.. currentmodule:: caiman.utils.visualization

.. autofunction:: plot_contours
.. autofunction:: view_patches_bar
.. autofunction:: view_patches
.. autofunction:: nb_view_patches
.. autofunction:: nb_imshow
.. autofunction:: nb_plot_contour

Cluster
-------
.. currentmodule:: caiman.cluster

.. autofunction:: start_server
.. autofunction:: stop_server

Utilities
---------------
.. currentmodule:: caiman.source_extraction.cnmf.utilities


.. autofunction:: manually_refine_components
.. autofunction:: order_components		  
.. autofunction:: extract_DF_F

.. currentmodule:: caiman.base.movies

.. autoclass:: movie