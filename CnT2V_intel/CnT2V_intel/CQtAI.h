/*************************************************************

    程序名称:基于Qt线程类的AI类
    程序版本:REV 0.1
    创建日期:20240307
    设计编写:王祥福
    作者邮箱:rainhenry@savelife-tech.com

    版本修订
        REV 0.1   20240307      王祥福    创建文档

*************************************************************/
//------------------------------------------------------------
//  重定义保护
#ifndef __CQTAI_H__
#define __CQTAI_H__

//------------------------------------------------------------
//  包含头文件
#include <QString>
#include <QVariant>
#include <QThread>
#include <QSemaphore>
#include <QMutex>
#include <QElapsedTimer>

#include "CPyAI.h"

//------------------------------------------------------------
//  类定义
class CQtAI: public QThread
{
    Q_OBJECT

public:
    //  状态枚举
    typedef enum
    {
        EAISt_Ready = 0,
        EAISt_Busy,
        EAISt_Error
    }EAISt;

    //  命令枚举
    typedef enum
    {
        EAIcmd_Null = 0,
        EAIcmd_ExTranslateCn2En,         //  中文到英文的翻译命令
        EAIcmd_ExConvIR,                 //  执行转换模型，并编译
        EAIcmd_ExTextToVideo_iGPU_NPU,   //  使用iGPU+NPU进行加速推理文字生成视频
        EAIcmd_ExTextToVideo_NoAcc,      //  不使用加速器进行文字生成视频的推理
        EAIcmd_Release,                  //  退出释放
    }EAIcmd;

    //  构造和析构函数
    CQtAI();
    ~CQtAI() override;

    //  线程运行本体
    void run() override;

    //  初始化
    void Init(void);

    //  执行一次翻译
    void ExTranslateCn2En(QString prompt);

    //  删除本地已有的IR模型
    void DeleteLocalIRmodel(void);

    //  获取已有IR模型的参数
    //  成功返回0，其他值表示错误
    int GetLocalIRargs(CPyAI::SInferenceArgs* p_args_out);

    //  执行一次导出IR模型并编译模型
    void ExConvIR(CPyAI::SInferenceArgs args);

    //  IR模型是否可用
    //  可用返回true,不可用返回false
    bool IR_model_is_valid(void);

    //  执行英文文本到视频文件的生成(使用iGPU+NPU加速器)
    void Text_To_Video_with_iGPU_NPU(
        QString prompt,                      //  输入的英文文本
        int steps,                           //  推理步数
        QString out_gif_file,                //  输出的gif动图文件
        QString out_mp4_file                 //  输出的mp4视频文件
        );

    //  执行英文文本到视频文件的生成(未使用加速器加速器)
    void Text_To_Video_NoAcc(
        QString prompt,                      //  输入的英文文本
        int steps,                           //  推理步数
        CPyAI::SInferenceArgs args,          //  推理参数
        QString out_gif_file,                //  输出的gif动图文件
        QString out_mp4_file                 //  输出的mp4视频文件
        );

    //  得到当前状态
    EAISt GetStatus(void);

    //  释放资源
    void Release(void);

signals:
    //  环境就绪
    void send_environment_ready(void);

    //  当完成翻译
    void send_translate_cn2en_finish(QString out_text, qint64 run_time_ns);

    //  当导出IR模型完成
    void send_exportIR_finish(qint64 run_time_ns);

    //  当完成文字生成视频
    void send_text_to_video_finish(qint64 run_time_ns, bool with_iGPU_NPU);

private:
    //  内部私有变量
    QSemaphore sem_cmd;                    //  命令信号量
    QMutex cmd_mutex;                      //  命令互斥量
    EAISt cur_st;                          //  当前状态
    EAIcmd cur_cmd;                        //  当前命令
    CPyAI* py_ai;                          //  Python的AI对象

    //  翻译相关
    QString tsl_prompt;                    //  翻译的输入

    //  导出IR模型相关
    CPyAI::SInferenceArgs export_args;     //  导出模型参数

    //  文字生成视频的推理相关
    QString ttv_prompt;                    //  生成视频的输入提示词
    int ttv_steps;                         //  生成视频的推理步数
    CPyAI::SInferenceArgs ttv_args;        //  生成视频的推理参数(仅非加速器使用)
    QString ttv_out_gif_file;              //  生成视频的输出gif动图文件
    QString ttv_out_mp4_file;              //  生成视频的输出mp4视频文件
};

//------------------------------------------------------------
#endif  //  __CQTAI_H__



