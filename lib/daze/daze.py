import warnings, numpy as np
from itertools import product
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_matplotlib_support
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import unique_labels
from sklearn.base import is_classifier
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .measures import (
    _MEASURE_NAMES, _MEASURES_DICT,
    _ColumnMeasure, _RowMeasure,
    _SummaryMeasure, _AverageMeasure,
    Count
)

def _normalize(cm, method):
    with np.errstate(all='ignore'):
        if method == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif method == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif method == 'all':
            cm = cm / cm.sum()
    return np.nan_to_num(cm)

class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.

    It is recommend to use :func:`plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.

    Parameters
    ----------
    confusion_matrix : :class:`numpy:numpy.ndarray` of shape (n_classes, n_classes)
        Confusion matrix.

    display_labels : :class:`numpy:numpy.ndarray` of shape (n_classes,), default=None
        Display labels for plot. If None, display labels are set from 0 to `n_classes - 1`.

    Attributes
    ----------
    im_ : :class:`matplotlib:matplotlib.image.AxesImage`
        Image representing the confusion matrix.

    text_ : :class:`numpy:numpy.ndarray` `(dtype=`:class:`matplotlib:matplotlib.text.Text`) of shape (n_classes, n_classes), or None
        Array of matplotlib axes. `None` if `include_values` is false.

    ax_ : :class:`matplotlib:matplotlib.axes.Axes`
        Axes with confusion matrix.

    figure_ : :class:`matplotlib:matplotlib.figure.Figure`
        Figure containing the confusion matrix.

    See Also
    --------
    plot_confusion_matrix : Plot Confusion Matrix.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import confusion_matrix
    >>> from daze import ConfusionMatrixDisplay
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> clf = SVC(random_state=0)
    >>> clf.fit(X_train, y_train)
    SVC(random_state=0)
    >>> predictions = clf.predict(X_test)
    >>> cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    >>> disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    >>> disp.plot() # doctest: +SKIP
    """
    @_deprecate_positional_args
    def __init__(self, confusion_matrix, *, display_labels=None):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    @_deprecate_positional_args
    def plot(self, *,
             include_measures=True, measures=('a', 'c', 'p', 'r', 'f1'),
             measures_format=None, include_summary=True, summary_type='macro',
             include_values=True, values_format=None,
             cmap='viridis', xticks_rotation='horizontal',
             ax=None, colorbar=True, normalize=None):
        """Plot visualization.

        Parameters
        ----------
        include_measures : bool, default=True
            Includes measures outside the confusion matrix.

        measures : array-like of str
            | Controls which measures to display outside the confusion matrix.
            | Can contain any of:

            - `'a'` for accuracy,
            - `'c'` for row/column counts,
            - `'tp'` for true positives,
            - `'fp'` for false positives,
            - `'tn'` for true negatives,
            - `'fn'` for false negatives,
            - `'tpr'` for true positive rate,
            - `'fpr'` for false positive rate,
            - `'tnr'` for true negative rate,
            - `'fnr'` for false negative rate,
            - `'p'` for precision,
            - `'r'` for recall (same as `'tpr'`),
            - `'f1'` for F1 score.

            Defaults to `('a', 'c', 'p', 'r', 'f1')`.

        measures_format : str, default=None
            Format specification for values in confusion matrix. If `None`, the format specification is `'.3f'`.

        include_summary : bool, default=True
            Includes summary values in the corner above the confusion matrix.

        summary_type : {'micro', 'macro'}, default='macro'
            Type of averaging used for summary measures.

        include_values : bool, default=True
            Includes values in confusion matrix.

        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is `'d'` or `'.2g'` whichever is shorter.

        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.

        xticks_rotation : {'vertical', 'horizontal'} or float, default='horizontal'
            Rotation of xtick labels.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is created.

        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.

        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If `None`, confusion matrix will not be normalized.

        Returns
        -------
        display : :class:`ConfusionMatrixDisplay`
        """
        check_matplotlib_support('ConfusionMatrixDisplay.plot')
        import matplotlib.pyplot as plt

        # if ax is None:
        #     fig, ax = plt.subplots()
        # else:
        #     fig = ax.figure
        ax = plt.gca() if ax is None else ax
        fig = ax.figure

        cm = self.confusion_matrix
        cm_disp = _normalize(cm, normalize)
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm_disp, interpolation='nearest', cmap=cmap)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_measures:
            self.text_ = np.empty((n_classes + 1, n_classes + 1), dtype=object)
        else:
            self.text_ = np.empty_like(cm, dtype=object)

        if include_values:
            # print text with appropriate color depending on background
            thresh = (cm_disp.max() + cm_disp.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm_disp[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm_disp[i, j], '.2g')
                    if cm_disp.dtype.kind != 'f':
                        text_d = format(cm_disp[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm_disp[i, j], values_format)

                self.text_[i + include_measures, j] = ax.text(j, i, text_cm, ha='center', va='center', color=color)

        if include_measures:
            measures_format = '.3f' if measures_format is None else measures_format
            fmt = lambda value: measures_format if isinstance(value, float) else 'd'

            # validate measures
            measures = list(measures)
            measures = list(dict.fromkeys(measures))
            if len(measures) == 0:
                raise ValueError('Expected at least one measure')
            if any(name not in _MEASURE_NAMES for name in measures):
                raise ValueError('Expected measures to be any combination of {}'.format(_MEASURE_NAMES))

            prediction_text, true_text = [[] for _ in range(n_classes)], [[] for _ in range(n_classes)]
            if include_summary:
                summary_text = []

            for name in measures:
                measure = _MEASURES_DICT[name]

                if issubclass(measure, _ColumnMeasure):
                    if issubclass(measure, Count):
                        values = measure(cm)(axis=0)
                        for i in range(n_classes):
                            value = format(values[i], fmt(values[i]))
                            prediction_text[i].append('{}: {}'.format(measure.label, value))
                    else:
                        values = measure(cm)()
                        for i in range(n_classes):
                            value = format(values[i], fmt(values[i]))
                            prediction_text[i].append('{}: {}'.format(measure.label, value))

                if issubclass(measure, _RowMeasure):
                    if issubclass(measure, Count):
                        values = measure(cm)(axis=1)
                        for i in range(n_classes):
                            value = format(values[i], fmt(values[i]))
                            true_text[i].append('{}: {}'.format(measure.label, value))
                    else:
                        values = measure(cm)()
                        for i in range(n_classes):
                            value = format(values[i], fmt(values[i]))
                            true_text[i].append('{}: {}'.format(measure.label, value))

                if include_summary:
                    if issubclass(measure, _SummaryMeasure):
                        if issubclass(measure, _AverageMeasure):
                            subscript = 'µ' if summary_type == 'micro' else 'M'
                            summary = measure(cm)(summary_type)
                            value = format(summary, fmt(summary))
                            summary_text.append(r'{}$_{}$: {}'.format(measure.label, subscript, value))
                        else:
                            summary = measure(cm)()
                            value = format(summary, fmt(summary))
                            summary_text.append(r'{}: {}'.format(measure.label, value))

            prediction_text = ['\n'.join(class_text) for class_text in prediction_text]
            true_text = ['\n'.join(class_text) for class_text in true_text]
            for i in range(n_classes):
                self.text_[0, i] = ax.text(i, -1, prediction_text[i], ha='center', va='center')
                self.text_[i, n_classes] = ax.text(n_classes, i, true_text[i], ha='center', va='center')

            if include_summary:
                summary_text = '\n'.join(summary_text)
                self.text_[0, n_classes] = ax.text(n_classes, -1, summary_text, ha='center', va='center')

        if colorbar:
            size = '8.5%'
            top = False
            divider = make_axes_locatable(ax)
            if include_measures:
                if any(issubclass(_MEASURES_DICT[measure], _RowMeasure) for measure in measures):
                    if any(issubclass(_MEASURES_DICT[measure], _ColumnMeasure) for measure in measures):
                        cax = divider.append_axes('bottom', size=size, pad=0.7)
                    else:
                        top = True
                        cax = divider.append_axes('top', size=size, pad=0.1)
                    cbar = {'cax': cax, 'orientation': 'horizontal'}
                else:
                    cax = divider.append_axes('right', size=size, pad=0.1)
                    cbar = {'cax': cax}
            else:
                cax = divider.append_axes('right', size=size, pad=0.1)
                cbar = {'cax': cax}
            c = fig.colorbar(self.im_, **cbar)
            if top:
                c.ax.xaxis.set_ticks_position('top')

        display_labels = np.arange(n_classes) if self.display_labels is None else self.display_labels
        ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
               xticklabels=display_labels, yticklabels=display_labels,
               ylabel='True label', xlabel='Predicted label')

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self

@_deprecate_positional_args
def plot_confusion_matrix(estimator, X=None, y_true=None, *,
                          labels=None, display_labels=None,
                          sample_weight=None, normalize=None,
                          include_measures=True, measures=('a', 'c', 'p', 'r', 'f1'),
                          measures_format=None, include_summary=True, summary_type='macro',
                          include_values=True, values_format=None,
                          xticks_rotation='horizontal', cmap='viridis',
                          ax=None, colorbar=True):
    """Plots a confusion matrix and annotates it with per-class and overall evaluation measures.

    Parameters
    ----------
    estimator : estimator instance or :class:`numpy:numpy.ndarray` of shape (n_classes, n_classes)
        Either:

        - Fitted classifier or a fitted :class:`sklearn:sklearn.pipeline.Pipeline` in which the last estimator is a classifier.
        - Pre-computed confusion matrix.

    X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None
        Input values. Only required if `estimator` is a classifier object.

    y_true : array-like of shape (n_samples,), default=None
        Target values. Only required if `estimator` is a classifier object.

    labels : array-like of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.

    display_labels : array-like of shape (n_classes,), default=None
        Target names used for plotting. By default, `labels` will be used if
        it is defined, otherwise the unique labels of `y_true` and `y_pred` will be used.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If `None`, confusion matrix will not be normalized.

    include_measures : bool, default=True
        Includes measures outside the confusion matrix.

    measures : array-like of str
            | Controls which measures to display outside the confusion matrix.
            | Can contain any of:

            - `'a'` for accuracy,
            - `'c'` for row/column counts,
            - `'tp'` for true positives,
            - `'fp'` for false positives,
            - `'tn'` for true negatives,
            - `'fn'` for false negatives,
            - `'tpr'` for true positive rate,
            - `'fpr'` for false positive rate,
            - `'tnr'` for true negative rate,
            - `'fnr'` for false negative rate,
            - `'p'` for precision,
            - `'r'` for recall (same as `'tpr'`),
            - `'f1'` for F1 score.

            Defaults to `('a', 'c', 'p', 'r', 'f1')`.

    measures_format : str, default=None
        Format specification for values in confusion matrix. If `None`, the format specification is `'.3f'`.

    include_summary : bool, default=True
        Includes summary values in the corner above the confusion matrix.

    summary_type : {'micro', 'macro'}, default='macro'
        Type of averaging used for summary measures.

    include_values : bool, default=True
        Includes values in confusion matrix.

    values_format : str, default=None
        Format specification for values in confusion matrix. If `None`,
        the format specification is `'d'` or `'.2g'` whichever is shorter.

    cmap : str or :class:`matplotlib:matplotlib.colors.Colormap`, default='viridis'
        Colormap recognized by matplotlib.

    xticks_rotation : {'vertical', 'horizontal'} or float, default='horizontal'
        Rotation of xtick labels.

    ax : :class:`matplotlib:matplotlib.axes.Axes`, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.

    colorbar : bool, default=True
        Whether or not to add a colorbar to the plot.

    Returns
    -------
    display : :class:`ConfusionMatrixDisplay`

    See Also
    --------
    ConfusionMatrixDisplay : Confusion Matrix visualization.

    Examples
    --------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> from daze import plot_confusion_matrix
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> clf = SVC(random_state=0)
    >>> clf.fit(X_train, y_train)
    SVC(random_state=0)
    >>> plot_confusion_matrix(clf, X_test, y_test)  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    """
    check_matplotlib_support('plot_confusion_matrix')

    if is_classifier(estimator):
        if X is None or y_true is None:
            raise ValueError('Expected input values X and true target values y_true to be set')
        y_pred = estimator.predict(X)
        cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, labels=labels, normalize=None)
    else:
        try:
            cm = np.array(estimator, dtype=int)
        except:
            raise ValueError('plot_confusion_matrix only supports classifiers (with input and target values) or a pre-computed un-normalized confusion matrix')
        if cm.ndim != 2:
            raise ValueError('plot_confusion_matrix expects a 2D array-like object as a confusion matrix')
        if cm.shape[0] != cm.shape[1]:
            raise ValueError('plot_confusion_matrix expects the confusion matrix to be square')
        if X is not None or y_true is not None:
            warnings.warn('Ignoring X and y_true, as a pre-computed confusion matrix was passed to plot_confusion_matrix', RuntimeWarning)

    if display_labels is None:
        display_labels = unique_labels(y_true, y_pred) if labels is None else labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    return disp.plot(include_measures=include_measures, measures=measures,
                     measures_format=measures_format, include_summary=include_summary, summary_type=summary_type,
                     include_values=include_values, values_format=values_format,
                     cmap=cmap, xticks_rotation=xticks_rotation,
                     ax=ax, colorbar=colorbar, normalize=normalize)