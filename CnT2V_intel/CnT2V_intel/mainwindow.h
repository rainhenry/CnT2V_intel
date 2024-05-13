#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QTimer>
#include <QPainter>
#include <QButtonGroup>

#include <QMediaPlayer>
#include <QVideoProbe>

#include "CQtAI.h"

//  文字生成视频的缓存文件
#define TEXT_TO_VIDEO_GIF_TMP_FILE    "tmp.gif"
#define TEXT_TO_VIDEO_MP4_TMP_FILE    "tmp.mp4"

//  最小的帧数目
#define TOTAL_FRAME_NUM_MIN           16

//  再次开始之前，是否删除缓存的视频文件
#define BEGIN_DELETE_VIDEO_TMP_FILE   0

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    //  视频的播放状态
    typedef enum
    {
        EVideoSt_Stop = 0,
        EVideoSt_Play,
        EVideoSt_Pause,
    }EVideoSt;

    //  加速器模式
    typedef enum
    {
        EAcceleratorMode_iGPU_NPU = 0,       //  iGPU+NPU混合加速
        EAcceleratorMode_NoAcc,              //  无加速器
    }EAcceleratorMode;

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    //  加速器模式单选组
    QButtonGroup acc_mode_grp;

    //  当前的加速器模式
    EAcceleratorMode cur_acc_mode;

    //  QtAI对象
    CQtAI qtai;

    //  定时器
    QTimer* timer;

    //  当前忙
    bool is_cur_busy;

    //  只进行翻译
    bool only_tsl_flag;

    //  只进行模型导出
    bool only_export_IR;

    //  视频播放
    QMediaPlayer* m_mediaplayer;

    //  视频播放状态
    EVideoSt video_st;

    //  视频探针
    QVideoProbe* m_videoprobe;

    //  进度字符串后缀
    QString progress_suffix_str;

    //  上一次的进度值
    int last_steps;

    //  关闭UI
    void Disable_UI(void);

    //  打开UI
    void Enable_UI(void);

    //  载入视频
    void LoadVideo(QString filename);


private slots:
    //  加速器模式单选槽
    void slot_AccModeSelect(int mode);

    //  当AI环境就绪
    void slot_OnAIEnvReady(void);

    //  当完成一次翻译
    void slot_OnTranslateCn2EnFinish(QString out_text, qint64 run_time_ns);

    //  当完成一次IR模型导出和编译
    void slot_OnExportIRFinish(qint64 run_time_ns);

    //  当完成一次文字生成视频
    void slot_OnTextToVideoFinish(qint64 run_time_ns, bool with_iGPU_NPU);

    //  定时器
    void slot_timeout();

    //  视频流探针槽
    void slot_OnVideoProbeFrame(const QVideoFrame& frame);


    //  单击只将中文翻译成英文的按钮
    void on_pushButton_tslonly_clicked();

    //  单击通过英文生成视频按钮
    void on_pushButton_byen_clicked();

    //  当单击只导出IR模型
    void on_pushButton_export_compile_clicked();

    //  单击视频播放按钮
    void on_pushButton_play_clicked();

    //  单击视频暂停按钮
    void on_pushButton_pause_clicked();

    //  单击视频停止按钮
    void on_pushButton_stop_clicked();

    //  单击通过中文生成视频按钮
    void on_pushButton_bycn_clicked();

    //  当视频宽度数值发生变化
    void on_spinBox_res_w_valueChanged(int arg1);

    //  当视频高度数值发生变化
    void on_spinBox_res_h_valueChanged(int arg1);

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
