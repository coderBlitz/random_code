//#include<QtGui>
#include<QtWidgets>
#include<iostream>
#include<string>

using std::cout;
using std::endl;
using std::string;

static QTextEdit *text;

void setContents();

void printContents(){
	// TextEdit->toPlainText() return a QString object. The extraction operator for QString is non-existant, so convert to std::string
	cout << "Cotents: \"" << text->toPlainText().toStdString() << "\"\n";
}

void setContents(){
	QString str ("Hello World");

	text->setPlainText(str);
}

int main(int argc, char *argv[]){
	QApplication app(argc, argv); // Setup the QT application. Necessary everytime

	text = new QTextEdit; // Initialize textEdit box
	QPushButton *debugButton = new QPushButton("Debug thing"); // Initialize button with text
	QPushButton *quitButton = new QPushButton("Bye");

	// Connect links the sender and its signal, to the reciever and its method/function
	QObject::connect(quitButton, SIGNAL(clicked()), qApp, SLOT(quit()) );
	// This version of connect allows an event to call another non-widget thing
	QObject::connect(debugButton, &QPushButton::clicked, printContents);

	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(text);
	layout->addWidget(debugButton);
	layout->addWidget(quitButton);

	QWidget window;
	window.setLayout(layout);

	window.show();

	return app.exec(); // Start the application event
}
