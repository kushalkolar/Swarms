from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from .pytemplates.mainwindow_pytemplate import Ui_MainWindow
from core import tracker, utils
import numpy as np
from .parameters import Parameters
from functools import partial
from .processing import *


class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self, parent=None)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.params = Parameters(
            diameter=self.ui.spinBoxDiameter.value(),
            minmass=self.ui.spinBoxMinmass.value(),
            maxmass=self.ui.spinBoxMaxmass.value(),
            maxsize=self.ui.doubleSpinBoxMaxsize.value(),
            search_range=self.ui.spinBoxLinkerSearchRange.value(),
            memory=self.ui.spinBoxLinkerMemory.value(),
            adjust_gamma=self.ui.checkBoxGamma.isChecked(),
            gamma=self.ui.doubleSpinBoxGamma.value(),
            use_clahe=self.ui.checkBoxCLAHE.isChecked(),
            clahe_clip_limit=self.ui.doubleSpinBoxCLAHEClipLimit.value(),
            clahe_grid_size=(self.ui.spinBoxGridSizeX.value(), self.ui.spinBoxGridSizeY.value()),
            circle_param1=self.ui.doubleSpinBoxCirclesParam1.value(),
            circle_param2=self.ui.spinBoxCirclesParam2.value(),
            circle_minradius=self.ui.spinBoxMaskMinRadius.value(),
            circle_maxradius=self.ui.spinBoxMaskMaxRadius.value(),
            use_neural_network=self.ui.checkBoxUseNeuralNetworkModel.isChecked(),
            neural_network_model_filepath=self.ui.lineEditNeuralNetworkModelPath.text(),
            params_name=self.ui.lineEditCurrentItemName.text()
        )
        #pg.setConfigOption('imageAxisOrder', 'row-major')

        self.ui.image_item.ui.menuBtn.hide()
        self.ui.image_item.ui.roiBtn.hide()

        self.video = np.ndarray

        self.connect_actions()
        self.connect_ui()

        self.arena_mask = None

    def connect_actions(self):
        self.ui.actionOpen_Video.triggered.connect(self.load_video)

    def connect_ui(self):
        self.ui.spinBoxDiameter.valueChanged.connect(self.update_params_from_ui)
        self.ui.spinBoxMinmass.valueChanged.connect(self.update_params_from_ui)
        self.ui.spinBoxMaxmass.valueChanged.connect(self.update_params_from_ui)
        self.ui.doubleSpinBoxMaxsize.valueChanged.connect(self.update_params_from_ui)
        self.ui.spinBoxLinkerSearchRange.valueChanged.connect(self.update_params_from_ui)
        self.ui.spinBoxLinkerMemory.valueChanged.connect(self.update_params_from_ui)
        self.ui.checkBoxGamma.clicked.connect(self.update_params_from_ui)
        self.ui.doubleSpinBoxGamma.valueChanged.connect(self.update_params_from_ui)
        self.ui.checkBoxCLAHE.clicked.connect(self.update_params_from_ui)
        self.ui.doubleSpinBoxCLAHEClipLimit.valueChanged.connect(self.update_params_from_ui)
        self.ui.spinBoxGridSizeX.valueChanged.connect(self.update_params_from_ui)
        self.ui.spinBoxGridSizeY.valueChanged.connect(self.update_params_from_ui)
        self.ui.doubleSpinBoxCirclesParam1.valueChanged.connect(self.update_params_from_ui)
        self.ui.spinBoxCirclesParam2.valueChanged.connect(self.update_params_from_ui)
        self.ui.spinBoxMaskMinRadius.valueChanged.connect(self.update_params_from_ui)
        self.ui.spinBoxMaskMaxRadius.valueChanged.connect(self.update_params_from_ui)

        self.ui.pushButtonRecreateMask.clicked.connect(self.create_arena_mask)

        self.ui.horizontalSliderFrameIndex.valueChanged.connect(self.update_params_from_ui)

        self.ui.pushButtonUpdateNow.clicked.connect(partial(self.update_params_from_ui, True))

    def create_arena_mask(self):
        self.arena_mask = utils.get_mask(self.video[0, :, :],
                                         param1=self.ui.doubleSpinBoxCirclesParam1.value(),
                                         param2=self.ui.spinBoxCirclesParam2.value(),
                                         min_radius=self.ui.spinBoxMaskMinRadius.value(),
                                         max_radius=self.ui.spinBoxMaskMaxRadius.value())

        self.update_params_from_ui()

    def update_params_from_ui(self, force=False):
        if not self.ui.pushButtonLiveUpdate.isChecked():
            if not force:
                return

        self.ui.image_item.setImage(self.video[self.ui.horizontalSliderFrameIndex.value(), :, :])

        self.ui.statusbar.showMessage('Updating params, please wait ...')

        self.params = Parameters(
            diameter=self.ui.spinBoxDiameter.value(),
            minmass=self.ui.spinBoxMinmass.value(),
            maxmass=self.ui.spinBoxMaxmass.value(),
            maxsize=self.ui.doubleSpinBoxMaxsize.value(),
            search_range=self.ui.spinBoxLinkerSearchRange.value(),
            memory=self.ui.spinBoxLinkerMemory.value(),
            adjust_gamma=self.ui.checkBoxGamma.isChecked(),
            gamma=self.ui.doubleSpinBoxGamma.value(),
            use_clahe=self.ui.checkBoxCLAHE.isChecked(),
            clahe_clip_limit=self.ui.doubleSpinBoxCLAHEClipLimit.value(),
            clahe_grid_size=(self.ui.spinBoxGridSizeX.value(), self.ui.spinBoxGridSizeY.value()),
            circle_param1=self.ui.doubleSpinBoxCirclesParam1.value(),
            circle_param2=self.ui.spinBoxCirclesParam2.value(),
            circle_minradius=self.ui.spinBoxMaskMinRadius.value(),
            circle_maxradius=self.ui.spinBoxMaskMaxRadius.value(),
            use_neural_network=self.ui.checkBoxUseNeuralNetworkModel.isChecked(),
            neural_network_model_filepath=self.ui.lineEditNeuralNetworkModelPath.text(),
            params_name=self.ui.lineEditCurrentItemName.text()
        )

        annotated = process_frame(self.video[self.ui.horizontalSliderFrameIndex.value(), :, :], self.params, self.arena_mask)

        self.ui.image_item.setImage(np.swapaxes(annotated, 0, 1))
        print(annotated.shape)

        self.ui.statusbar.showMessage('Finished updating params!')

    def load_video(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose video file', '', '(*.avi)')
        if path == '':
            return None
        if path[0] == '':
            return None
        vid_name = path[0].split('/')[-1]
        self.ui.labelVideoName.setText(vid_name)
        self.ui.lineEditCurrentItemName.setText(vid_name)

        self.ui.statusbar.showMessage('Loading video, please wait...')
        self.video = utils.load_video(path[0])

        self.ui.horizontalSliderFrameIndex.setMaximum(self.video.shape[0])
        self.ui.spinBoxFrameIndex.setMaximum(self.video.shape[0] - 1)
        self.ui.horizontalSliderFrameIndex.setValue(0)

        self.ui.image_item.setImage(self.video[0, :, :].T)
        self.arena_mask = None
        self.ui.statusbar.showMessage('Finished loading video!')