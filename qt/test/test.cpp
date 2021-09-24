#include<QtWidgets>
#include<iostream>
#include<string>
#include<sstream>

static QVBoxLayout *layout;

static int count = 1;
void addButton(){
	std::ostringstream convert;

	convert << "Test" << count++;
	QString name = QString::fromStdString(convert.str());

	QPushButton *newButt = new QPushButton(name);
	layout->addWidget(newButt);
}

int main(int argc, char *argv[]){
	QApplication app(argc, argv);

	QPushButton *push = new QPushButton("Test");

	QObject::connect(push, &QPushButton::clicked, addButton);

	layout = new QVBoxLayout();
	layout->addWidget(push);

	QWidget window;
	window.setLayout(layout);

	window.show();

	return app.exec();
}
