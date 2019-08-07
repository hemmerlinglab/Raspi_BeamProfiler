
import sys
from PyQt5.QtWidgets import QSizePolicy, QTextEdit, QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout,QPushButton, QHBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import numpy as np
import scipy
from lmfit import Minimizer, Parameters, report_fit

import random

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


# define objective function: returns the array to be minimized
def fcn2min(params, x, data, plot_fit = False):
    """Model a decaying sine wave and subtract data."""
    amplitude = params['amplitude']
    waist = params['waist']
    x_offset = params['x_offset']
    y_offset = params['y_offset']
    
    if plot_fit == False:
        model = amplitude/2.0 * (1 - scipy.special.erf(np.sqrt(2.0) * (x - x_offset)/waist)) + y_offset
        
        return model - data
    else:
        x_plot = np.linspace(np.min(x), np.max(x), 100)
        model = amplitude/2.0 * (1 - scipy.special.erf(np.sqrt(2.0) * (x_plot - x_offset)/waist)) + y_offset
        return (x_plot, model)
    


class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 table - pythonspot.com'
        self.left = 0
        self.top = 0
        self.width = 1000
        self.height = 500
        self.no_of_rows = 20
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        self.createTable()
 
        self.button = QPushButton('Fit', self)
        self.button.setToolTip('This is an example button')
        self.button.move(100,70)
        self.button.clicked.connect(self.button_click)


        self.canvas = PlotCanvas(self, width=5, height=4)
        self.canvas.move(0,0)

        self.textbox = QTextEdit()

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.tableWidget) 
        self.layout.addWidget(self.canvas) 
        self.layout.addWidget(self.button) 
        self.layout.addWidget(self.textbox) 
        self.setLayout(self.layout) 
 
        # Show widget
        self.show()
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
      
        self.show()

        
    @pyqtSlot()
    def button_click(self):
        print('PyQt5 button click')
        #self.tableWidget.setItem(0,0, QTableWidgetItem("pressed"))

        self.x = np.array([])
        self.y = np.array([])

        for k in range(self.no_of_rows):
            
            hlp = self.tableWidget.item(k,0)
            if not hlp is None:
                self.x = np.append(self.x, np.float(hlp.text()))
            else:
                break
            hlp = self.tableWidget.item(k,1)
            if not hlp is None:
                self.y = np.append(self.y, np.float(hlp.text()))

        print(self.x)
        print(self.y)

        params = Parameters()
        params.add('amplitude', value=np.max(self.y), min=(np.max(self.y) - np.min(self.y))/2.0, max=(np.max(self.y) - np.min(self.y)))
        params.add('waist', value=(np.max(self.x)-np.min(self.x))/2.0, min=10.0, max=2000)
        params.add('x_offset', value=np.mean(self.x), min=np.min(self.x), max = np.max(self.x))
        params.add('y_offset', value=0.0, min=0.00, max=np.max(self.y), vary = False)

        # do fit, here with leastsq model
        minner = Minimizer(fcn2min, params, fcn_args=(self.x, self.y))
        result = minner.minimize()

        # write error report
        self.textbox.setText("")
        for k in params.keys():
            my_str = str(result.params[k].value)
            self.textbox.append(str(k) + " = " + my_str + "\n")

        self.canvas.x = self.x
        self.canvas.y = self.y

        self.canvas.plot(fit_plot = result)


    def createTable(self):
       # Create table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(self.no_of_rows)
        self.tableWidget.setColumnCount(2)
        #self.tableWidget.setItem(0,0, QTableWidgetItem("Cell (1,1)"))
        #self.tableWidget.setItem(0,1, QTableWidgetItem("Cell (1,2)"))
        #self.tableWidget.setItem(1,0, QTableWidgetItem("Cell (2,1)"))
        #self.tableWidget.setItem(1,1, QTableWidgetItem("Cell (2,2)"))
        #self.tableWidget.setItem(2,0, QTableWidgetItem("Cell (3,1)"))
        #self.tableWidget.setItem(2,1, QTableWidgetItem("Cell (3,2)"))
        #self.tableWidget.setItem(3,0, QTableWidgetItem("Cell (4,1)"))
        #self.tableWidget.setItem(3,1, QTableWidgetItem("Cell (4,2)"))
        self.tableWidget.move(0,0)

        hlp = np.array([
           [ 1524,3.66 ], 
           [ 1651,3.5 ],
           [ 1676.4,3.17 ],
           [ 1701.8,2.53 ],
           [ 1727.2,1.71 ],
           [ 1752.6,0.87 ],
           [ 1778,0.32 ],
           [ 1803.4,0.1 ],
           [ 1828.8,0.016 ],
           [ 1854.2,0.001 ],
            ])
        self.x = hlp[:, 0]
        self.y = hlp[:, 1]

        for k in range(len(self.x)):

            self.tableWidget.setItem(k,0, QTableWidgetItem(str(self.x[k])))
            self.tableWidget.setItem(k,1, QTableWidgetItem(str(self.y[k])))

        #self.tableWidget.installEventFilters(self)

        # table selection change
        self.tableWidget.doubleClicked.connect(self.on_click)
 
    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())
    
    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.KeyPress and
            event.matches(QtGui.QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(Window, self).eventFilter(source, event)



class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.x = []
        self.y = []
        self.plot()
 
 
    def plot(self, fit_plot = None):
        ax = self.figure.add_subplot(111)
        # data
        ax.plot(self.x, self.y, 'ro')
        # fit
        if not fit_plot is None:
            (fit_x, fit_y) = fcn2min(fit_plot.params, self.x, None, plot_fit = True)
            ax.plot(fit_x, fit_y)
        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
