import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QTextEdit, QPushButton, QVBoxLayout, QWidget, QFileDialog, QPlainTextEdit
from io import StringIO
import contextlib
import os
import webbrowser

#os.chdir('/home/alam/INPE/ADENIRAN/ANALISE/')

class PythonIDE(QMainWindow):
    def __init__(self):
        super(PythonIDE, self).__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Python IDE')

        self.text_editor = QTextEdit(self)
        self.text_editor.setGeometry(10, 10, 780, 300)

        self.output_widget = QTextEdit(self)
        self.output_widget.setGeometry(10, 320, 780, 200)

        
        self.fileOpenButton = QPushButton('Open File',self)
        self.fileOpenButton.setGeometry(10,560,150,30)#(10, 470, 780, 30)
        #self.fileOpenButton.clicked.connect(self.open_PyFile)
        layout = QVBoxLayout()
        layout.addWidget(self.fileOpenButton)
        self.fileOpenButton.clicked.connect(self.open_PyFile)
        
        self.run_button = QPushButton('Read_file', self)
        self.run_button.setGeometry(170,560,150,30)#(10, 500, 780, 30)
        self.run_button.clicked.connect(self.read_file)
        
        self.run_button = QPushButton('Run', self)
        self.run_button.setGeometry(330,560,150,30)#(10, 530, 780, 30)
        self.run_button.clicked.connect(self.run_code)
        
        self.btn_fechar = QPushButton("Fechar", self)
        self.btn_fechar.setGeometry(490,560,150,30)#(10, 560, 780, 30)
        self.btn_fechar.clicked.connect(self.close)
    
        
        
    def run_code(self):
        code = self.text_editor.toPlainText()
        output_stream = StringIO()
        
        with contextlib.redirect_stdout(output_stream):
            try:
            	exec(open(file_name).read(), globals())#exec(code)
            except Exception as e:
                print(e)

        output = output_stream.getvalue()
        self.output_widget.setPlainText(output)
        
    def open_PyFile(self):
    	global file_name
    	options = QFileDialog.Options()
    	file_name, _ = QFileDialog.getOpenFileName(self,"Choose File", "","py (*.py)",options=options)
    	#dir_name = QFileDialog.getExistingDirectory(options=QFileDialog.DontUseNativeDialog)
    	from pathlib import Path
    	p=Path(file_name)
    	dir_name=str(p.parent)
    	print (file_name,dir_name)
    	os.chdir(dir_name+'/')
    	content=open(file_name).read()
    	self.text_editor.setPlainText(content)
        
    def read_file(self):
    	webbrowser.open(fileName)
    	

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ide = PythonIDE()
    ide.show()
    os.chdir('/home/alam/GUI')
    sys.exit(app.exec_())
    
