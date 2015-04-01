#include "MainWindow.h"

#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.resize(450, 500);
    mainWindow.setWindowTitle("Simple menu");
    mainWindow.show();
    return app.exec();
}
