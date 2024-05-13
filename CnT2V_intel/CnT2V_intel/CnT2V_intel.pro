QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    CPyAI.cpp \
    CQtAI.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    CPyAI.h \
    CQtAI.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

##  多媒体支持
QT+=multimedia multimediawidgets

##  下面的路径需要根据你的PC的环境进行修改python的路径
INCLUDEPATH+=/usr/include/python3.10
INCLUDEPATH+=/usr/local/lib/python3.10/dist-packages/numpy/core/include
LIBS+=-lpython3.10
