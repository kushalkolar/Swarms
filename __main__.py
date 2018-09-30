from PyQt5.QtWidgets import QApplication
from gui.tracker import TrackerWindow


if __name__ == '__main__':
    app = QApplication([])

    tw = TrackerWindow()
    tw.show()

    app.exec_()
