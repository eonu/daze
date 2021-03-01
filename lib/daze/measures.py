import numpy as np

class _Measure:
    def __init__(self, confusion_matrix):
        self.cm = confusion_matrix

class _ColumnMeasure(_Measure): pass
class _RowMeasure(_Measure): pass
class _SummaryMeasure(_Measure): pass
class _AverageMeasure(_SummaryMeasure):
    def _class(self):
        raise NotImplementedError

class Accuracy(_SummaryMeasure):
    """A summary measure that computes the categorical accuracy."""
    name, label = 'a', 'Acc'

    def __call__(self):
        """
        Returns
        -------
        accuracy: 0 ≤ float ≤ 1
            The categorical accuracy.
        """
        return TP(self.cm)().sum() / self.cm.sum()

class Count(_ColumnMeasure, _RowMeasure):
    """A row and column measure that computes the counts of rows/columns."""
    name, label = 'c', '#'

    def __call__(self, axis):
        """
        Parameters
        ----------
        axis: {0, 1}
            The axis of the confusion matrix to perform the counts over.

        Returns
        -------
        count: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)`
            The integer counts of each row/column.
        """
        return self.cm.sum(axis=axis)

class TP(_ColumnMeasure):
    """A column measure that computes the true positives of each class."""
    name, label = 'tp', 'TP'

    def __call__(self):
        """
        Returns
        -------
        tp: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)`
            The true positives for each class.
        """
        return np.diag(self.cm)

class FP(_ColumnMeasure):
    """A column measure that computes the false positives of each class."""
    name, label = 'fp', 'FP'

    def __call__(self):
        """
        Returns
        -------
        fp: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)`
            The false positives for each class.
        """
        return self.cm.sum(axis=0) - TP(self.cm)()

class FN(_RowMeasure):
    """A row measure that computes the false positives of each class."""
    name, label = 'fn', 'FN'

    def __call__(self):
        """
        Returns
        -------
        fn: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)`
            The false negatives for each class.
        """
        return self.cm.sum(axis=1) - TP(self.cm)()

class TN(_RowMeasure):
    """A row measure that computes the true negatives of each class."""
    name, label = 'tn', 'TN'

    def __call__(self):
        """
        Returns
        -------
        tn: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)`
            The true negatives for each class.
        """
        return self.cm.sum() - (TP(self.cm)() + FP(self.cm)() + FN(self.cm)())

class TPR(_AverageMeasure, _RowMeasure):
    """
    A row measure that computes the true positive rate of each class.
    Also an average/summary measure that can compute the micro and macro averaged true positive rate over all classes.
    """
    name, label = 'tpr', 'TPR'

    def _class(self):
        return TP(self.cm)() / self.cm.sum(axis=1)

    def __call__(self, measure_type=None):
        """
        Parameters
        ----------
        measure_type: {'micro', 'macro'}, default=None
            The averaging method. If `None`, then no averaging is done and per-class true positive rates are returned.

        Returns
        -------
        tpr: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)` or 0 ≤ float ≤ 1
            The true positive rates for each class (if `measure_type=None`), otherwise the micro/macro-averaged true positive rate.
        """
        if measure_type is None:
            return self._class()
        elif measure_type == 'micro':
            tp_sum, fn_sum = TP(self.cm)().sum(), FN(self.cm)().sum()
            return tp_sum / (tp_sum + fn_sum)
        elif measure_type == 'macro':
            return self._class().mean()

class FNR(_AverageMeasure, _RowMeasure):
    """
    A row measure that computes the false negative rate of each class.
    Also an average/summary measure that can compute the micro and macro averaged false negative rate over all classes.
    """
    name, label = 'fnr', 'FNR'

    def _class(self):
        return 1 - TPR(self.cm)()

    def __call__(self, measure_type=None):
        """
        Parameters
        ----------
        measure_type: {'micro', 'macro'}, default=None
            The averaging method. If `None`, then no averaging is done and per-class false negative rates are returned.

        Returns
        -------
        fnr: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)` or 0 ≤ float ≤ 1
            The false negative rates for each class (if `measure_type=None`), otherwise the micro/macro-averaged false negative rate.
        """
        if measure_type is None:
            return self._class()
        elif measure_type == 'micro':
            return 1 - TPR(self.cm)('micro')
        elif measure_type == 'macro':
            return self._class().mean()

class TNR(_AverageMeasure, _ColumnMeasure):
    """
    A column measure that computes the true negative rate of each class.
    Also an average/summary measure that can compute the micro and macro averaged true negative rate over all classes.
    """
    name, label = 'tnr', 'TNR'

    def _class(self):
        tn, fp = TN(self.cm)(), FP(self.cm)()
        return tn / (tn + fp)

    def __call__(self, measure_type=None):
        """
        Parameters
        ----------
        measure_type: {'micro', 'macro'}, default=None
            The averaging method. If `None`, then no averaging is done and per-class true negative rates are returned.

        Returns
        -------
        tnr: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)` or 0 ≤ float ≤ 1
            The true negative rates for each class (if `measure_type=None`), otherwise the micro/macro-averaged true negative rate.
        """
        if measure_type is None:
            return self._class()
        elif measure_type == 'micro':
            tn_sum, fp_sum = TN(self.cm)().sum(), FP(self.cm)().sum()
            return tn_sum / (tn_sum + fp_sum)
        elif measure_type == 'macro':
            return self._class().mean()

class FPR(_AverageMeasure, _ColumnMeasure):
    """
    A column measure that computes the false positive rate of each class.
    Also an average/summary measure that can compute the micro and macro averaged false positive rate over all classes.
    """
    name, label = 'fpr', 'FPR'

    def _class(self):
        return 1 - TNR(self.cm)()

    def __call__(self, measure_type=None):
        """
        Parameters
        ----------
        measure_type: {'micro', 'macro'}, default=None
            The averaging method. If `None`, then no averaging is done and per-class false positive rates are returned.

        Returns
        -------
        fpr: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)` or 0 ≤ float ≤ 1
            The false positive rates for each class (if `measure_type=None`), otherwise the micro/macro-averaged false positive rate.
        """
        if measure_type is None:
            return self._class()
        elif measure_type == 'micro':
            return 1 - TNR(self.cm)('micro')
        elif measure_type == 'macro':
            return self._class().mean()

class Precision(_AverageMeasure, _ColumnMeasure):
    """
    A column measure that computes the precision of each class.
    Also an average/summary measure that can compute the micro and macro averaged precision over all classes.
    """
    name, label = 'p', 'P'

    def _class(self):
        return TP(self.cm)() / self.cm.sum(axis=0)

    def __call__(self, measure_type=None):
        """
        Parameters
        ----------
        measure_type: {'micro', 'macro'}, default=None
            The averaging method. If `None`, then no averaging is done and per-class precisions are returned.

        Returns
        -------
        precision: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)` or 0 ≤ float ≤ 1
            The precision for each class (if `measure_type=None`), otherwise the micro/macro-averaged precision.
        """
        if measure_type is None:
            return self._class()
        elif measure_type == 'micro':
            tp_sum, fp_sum = TP(self.cm)().sum(), FP(self.cm)().sum()
            return tp_sum / (tp_sum + fp_sum)
        elif measure_type == 'macro':
            return self._class().mean()

class Recall(TPR):
    """
    A row measure that computes the recall of each class.
    Also an average/summary measure that can compute the micro and macro averaged recall over all classes.

    .. seealso:: Equivalent to :class:`~TPR`.
    """
    name, label = 'r', 'R'

class F1(_AverageMeasure, _ColumnMeasure, _RowMeasure):
    """
    A row and column measure that computes the F1 score of each class.
    Also an average/summary measure that can compute the micro and macro averaged F1 score over all classes.
    """
    name, label = 'f1', 'F1'

    def _class(self):
        precision, recall = Precision(self.cm)(), Recall(self.cm)()
        return 2. * precision * recall / (precision + recall)

    def __call__(self, measure_type=None):
        """
        Parameters
        ----------
        measure_type: {'micro', 'macro'}, default=None
            The averaging method. If `None`, then no averaging is done and per-class F1 scores are returned.

        Returns
        -------
        f1: :class:`numpy:numpy.ndarray` `(dtype=int)` of shape `(n_classes,)` or 0 ≤ float ≤ 1
            The F1 score for each class (if `measure_type=None`), otherwise the micro/macro-averaged F1 score.
        """
        if measure_type is None:
            return self._class()
        elif measure_type == 'micro':
            precision, recall = Precision(self.cm)('micro'), Recall(self.cm)('micro')
            return 2. * precision * recall / (precision + recall)
            pass
        elif measure_type == 'macro':
            return self._class().mean()

_MEASURES = (Accuracy, Count, TP, FP, TN, FN, TPR, FPR, TNR, FNR, Precision, Recall, F1)
_MEASURE_NAMES = tuple(measure.name for measure in _MEASURES)
_MEASURES_DICT = {measure.name:measure for measure in _MEASURES}