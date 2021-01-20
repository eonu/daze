.. _measures:

``daze.measures``
=================

This module contains classes that allow for various confusion matrix evaluation measures to be computed.

All classes must be initialized with a `confusion_matrix` in the form of a square :class:`numpy:numpy.ndarray` .

Types of measures
-----------------

To ensure that multiple evaluation measures can be displayed alongside the confusion matrix without obstruction,
they are divided into three types of measures --- **column**, **row** and **summary**:

+---------------------------------------------------+-------+-----------+---------+------+----------+
| Measure & reference                               | Label | Specifier | Column? | Row? | Summary? |
+===================================================+=======+===========+=========+======+==========+
| Accuracy (:class:`~daze.measures.Accuracy`)       | `Acc` | ``'a'``   | ✗       | ✗    | ✔        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| Count (:class:`~daze.measures.Count`)             | `#`   | ``'c'``   | ✔       | ✔    | ✗        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| True Positives (:class:`~daze.measures.TP`)       | `TP`  | ``'tp'``  | ✔       | ✗    | ✗        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| False Positives (:class:`~daze.measures.FP`)      | `FP`  | ``'fp'``  | ✔       | ✗    | ✗        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| True Negatives (:class:`~daze.measures.TN`)       | `TN`  | ``'tn'``  | ✗       | ✔    | ✗        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| False Negatives (:class:`~daze.measures.FN`)      | `FN`  | ``'fn'``  | ✗       | ✔    | ✗        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| True Positive Rate (:class:`~daze.measures.TPR`)  | `TPR` | ``'tpr'`` | ✗       | ✔    | ✔        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| False Negative Rate (:class:`~daze.measures.FNR`) | `FNR` | ``'fnr'`` | ✗       | ✔    | ✔        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| True Negative Rate (:class:`~daze.measures.TNR`)  | `TNR` | ``'tnr'`` | ✔       | ✗    | ✔        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| False Positive Rate (:class:`~daze.measures.FPR`) | `FPR` | ``'fpr'`` | ✔       | ✗    | ✔        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| Precision (:class:`~daze.measures.Precision`)     | `P`   | ``'p'``   | ✔       | ✗    | ✔        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| Recall (:class:`~daze.measures.Recall`)           | `R`   | ``'r'``   | ✗       | ✔    | ✔        |
+---------------------------------------------------+-------+-----------+---------+------+----------+
| :math:`F_1` Score (:class:`~daze.measures.F1`)    | `F1`  | ``'f1'``  | ✔       | ✔    | ✔        |
+---------------------------------------------------+-------+-----------+---------+------+----------+

Note that the allocation of measures to the column and row categories is somewhat arbitrary, but still maintains some level of reason.

All summary measures apart from accuracy are displayed as a macro (`M`) or micro (:math:`\mu`) averaged quantity over the per-class measures,
indicated by a subscript `M` or :math:`\mu`.

These measures are displayed in the following way:

.. image:: /_static/confusion.svg
    :alt: Confusion Matrix Layout
    :width: 40%
    :align: center

Accuracy (``Accuracy``)
-----------------------

.. autoclass:: daze.measures.Accuracy
    :members: __call__

Count (``Count``)
-----------------

.. autoclass:: daze.measures.Count
    :members: __call__

True Positives (``TP``)
-----------------------

.. autoclass:: daze.measures.TP
    :members: __call__

False Positives (``FP``)
------------------------

.. autoclass:: daze.measures.FP
    :members: __call__

False Negatives (``FN``)
------------------------

.. autoclass:: daze.measures.FN
    :members: __call__

True Negatives (``TN``)
-----------------------

.. autoclass:: daze.measures.TN
    :members: __call__

True Positive Rate (``TPR``)
----------------------------

.. autoclass:: daze.measures.TPR
    :members: __call__

False Negative Rate (``FNR``)
-----------------------------

.. autoclass:: daze.measures.FNR
    :members: __call__

True Negative Rate (``TNR``)
----------------------------

.. autoclass:: daze.measures.TNR
    :members: __call__

False Positive Rate (``FPR``)
-----------------------------

.. autoclass:: daze.measures.FPR
    :members: __call__

Precision (``Precision``)
-------------------------

.. autoclass:: daze.measures.Precision
    :members: __call__

Recall (``Recall``)
-------------------

.. autoclass:: daze.measures.Recall
    :members: __call__

:math:`F_1` Score (``F1``)
--------------------------

.. autoclass:: daze.measures.F1
    :members: __call__