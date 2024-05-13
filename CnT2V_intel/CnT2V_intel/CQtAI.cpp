/*************************************************************

    程序名称:基于Qt线程类的AI类
    程序版本:REV 0.1
    创建日期:20240307
    设计编写:王祥福
    作者邮箱:rainhenry@savelife-tech.com

    版本修订
        REV 0.1   20240307      王祥福    创建文档

*************************************************************/

//  包含头文件
#include "CQtAI.h"

//  步骤调试
void debug_steps_print(int steps)
{
    qDebug("------- steps=%d ----------", steps);
}

//  全局变量，用于步骤更新
int g_steps = 0;

//  构造函数
CQtAI::CQtAI():
    sem_cmd(0)
{
    //  初始化私有数据
    cur_st                   = EAISt_Ready;
    cur_cmd                  = EAIcmd_Null;
    ttv_steps                = 0;
    export_args.args_valid   = false;
    export_args.width        = 0;
    export_args.height       = 0;
    export_args.total_frames = 0;
    ttv_args.args_valid      = false;
    ttv_args.width           = 0;
    ttv_args.height          = 0;
    ttv_args.total_frames    = 0;
    

    //  创建Python的Chat对象
    py_ai = new CPyAI();
}

//  析构函数
CQtAI::~CQtAI()
{
    Release();   //  通知线程退出
    this->msleep(1000);
    this->quit();
    this->wait(500);

    delete py_ai;
}

//  初始化
void CQtAI::Init(void)
{
    //  暂时没有
}

void CQtAI::run()
{
    //  初始化python环境
    py_ai->Init();

    //  通知运行环境就绪
    emit send_environment_ready();

    //  现成主循环
    while(1)
    {
        //  获取信号量
        sem_cmd.acquire();

        //  获取当前命令和数据
        cmd_mutex.lock();
        EAIcmd now_cmd                        = this->cur_cmd;
        std::string tsl_prompt_str            = this->tsl_prompt.toStdString();
        std::string ttv_prompt_str            = this->ttv_prompt.toStdString();
        CPyAI::SInferenceArgs now_export_args = this->export_args;
        CPyAI::SInferenceArgs now_ttv_args    = this->ttv_args;
        int ttv_steps_val                     = this->ttv_steps;
        std::string ttv_out_gif_file_str      = this->ttv_out_gif_file.toStdString();
        std::string ttv_out_mp4_file_str      = this->ttv_out_mp4_file.toStdString();
        cmd_mutex.unlock();

        //  当为空命令
        if(now_cmd == EAIcmd_Null)
        {
            //  释放CPU
            this->sleep(1);
        }
        //  当为退出命令
        else if(now_cmd == EAIcmd_Release)
        {
            py_ai->Release();
            qDebug("Thread is exit!!");
            return;
        }
        //  当为中文翻译成英文命令
        else if(now_cmd == EAIcmd_ExTranslateCn2En)
        {
            //  设置忙
            cmd_mutex.lock();
            this->cur_st = EAISt_Busy;
            cmd_mutex.unlock();

            //  执行
            QString re_str;
            std::string std_str;
            QElapsedTimer run_time;
            run_time.start();
            std_str = py_ai->Translate_Cn2En_Ex(tsl_prompt_str.c_str());
            qint64 run_time_ns = run_time.nsecsElapsed();

            //  转换字符串格式
            re_str = std_str.c_str();

            //  发出操作完成
            emit send_translate_cn2en_finish(re_str, run_time_ns);

            //  完成处理
            cmd_mutex.lock();
            this->cur_st = EAISt_Ready;
            now_cmd = EAIcmd_Null;
            cmd_mutex.unlock();
        }
        //  当为转换模型的命令
        else if(now_cmd == EAIcmd_ExConvIR)
        {
            //  设置忙
            cmd_mutex.lock();
            this->cur_st = EAISt_Busy;
            cmd_mutex.unlock();

            //  执行
            QElapsedTimer run_time;
            run_time.start();
            py_ai->ConvModeltoIRformat_for_iGPU_NPU(now_export_args);
            qint64 run_time_ns = run_time.nsecsElapsed();

            //  发出操作完成
            emit send_exportIR_finish(run_time_ns);

            //  完成处理
            cmd_mutex.lock();
            this->cur_st = EAISt_Ready;
            now_cmd = EAIcmd_Null;
            cmd_mutex.unlock();
        }
        //  当为文字生成视频命令(iGPU+NPU加速器)
        else if(now_cmd == EAIcmd_ExTextToVideo_iGPU_NPU)
        {
            //  设置忙
            cmd_mutex.lock();
            this->cur_st = EAISt_Busy;
            cmd_mutex.unlock();

            //  执行
            QElapsedTimer run_time;
            run_time.start();
            py_ai->Text_To_Video_with_iGPU_NPU(
                       ttv_prompt_str.c_str(),
                       ttv_steps_val,
                       ttv_out_gif_file_str.c_str(),
                       ttv_out_mp4_file_str.c_str()
                      );
            qint64 run_time_ns = run_time.nsecsElapsed();

            //  发出操作完成
            emit send_text_to_video_finish(run_time_ns, true);

            //  完成处理
            cmd_mutex.lock();
            this->cur_st = EAISt_Ready;
            now_cmd = EAIcmd_Null;
            cmd_mutex.unlock();
        }
        //  当为文字生成视频命令(无加速器)
        else if(now_cmd == EAIcmd_ExTextToVideo_NoAcc)
        {
            //  设置忙
            cmd_mutex.lock();
            this->cur_st = EAISt_Busy;
            cmd_mutex.unlock();

            //  执行
            QElapsedTimer run_time;
            run_time.start();
            py_ai->Text_To_Video_NoAcc(
                       ttv_prompt_str.c_str(),
                       ttv_steps_val,
                       now_ttv_args,
                       ttv_out_gif_file_str.c_str(),
                       ttv_out_mp4_file_str.c_str()
                      );
            qint64 run_time_ns = run_time.nsecsElapsed();

            //  发出操作完成
            emit send_text_to_video_finish(run_time_ns, false);

            //  完成处理
            cmd_mutex.lock();
            this->cur_st = EAISt_Ready;
            now_cmd = EAIcmd_Null;
            cmd_mutex.unlock();
        }
        //  非法命令
        else
        {
            //  释放CPU
            QThread::sleep(1);
            qDebug("Unknow cmd code!!");
        }
    }
}

CQtAI::EAISt CQtAI::GetStatus(void)
{
    EAISt re;
    cmd_mutex.lock();
    re = this->cur_st;
    cmd_mutex.unlock();
    return re;
}

//  执行一次翻译
void CQtAI::ExTranslateCn2En(QString prompt)
{
    if(GetStatus() == EAISt_Busy) return;

    cmd_mutex.lock();
    this->cur_cmd = EAIcmd_ExTranslateCn2En;
    this->tsl_prompt = prompt;
    this->cur_st = EAISt_Busy;    //  设置忙
    cmd_mutex.unlock();

    sem_cmd.release();
}

//  删除本地已有的IR模型
void CQtAI::DeleteLocalIRmodel(void)
{
    std::string cmd;
    cmd = "rm -rf ";
    cmd += CPYAI_EXPORT_IR_MODEL_PATH;
    system(cmd.c_str());
}

//  获取已有IR模型的参数
//  成功返回0，其他值表示错误
int CQtAI::GetLocalIRargs(CPyAI::SInferenceArgs* p_args_out)
{
    return py_ai->GetLocalIRargs(p_args_out);
}

//  IR模型是否可用
//  可用返回true,不可用返回false
bool CQtAI::IR_model_is_valid(void)
{
    return py_ai->IR_model_is_valid();
}

//  执行一次导出IR模型并编译模型
void CQtAI::ExConvIR(CPyAI::SInferenceArgs args)
{
    if(GetStatus() == EAISt_Busy) return;

    cmd_mutex.lock();
    this->cur_cmd = EAIcmd_ExConvIR;
    this->export_args = args;
    this->cur_st = EAISt_Busy;    //  设置忙
    cmd_mutex.unlock();

    sem_cmd.release();
}

//  执行英文文本到视频文件的生成(使用iGPU+NPU加速器)
void CQtAI::Text_To_Video_with_iGPU_NPU(
    QString prompt,                      //  输入的英文文本
    int steps,                           //  推理步数
    QString out_gif_file,                //  输出的gif动图文件
    QString out_mp4_file                 //  输出的mp4视频文件
    )
{
    if(GetStatus() == EAISt_Busy) return;

    cmd_mutex.lock();
    this->cur_cmd = EAIcmd_ExTextToVideo_iGPU_NPU;
    this->ttv_prompt = prompt;
    this->ttv_steps = steps;
    this->ttv_out_gif_file = out_gif_file;
    this->ttv_out_mp4_file = out_mp4_file;
    this->cur_st = EAISt_Busy;    //  设置忙
    cmd_mutex.unlock();

    sem_cmd.release();
}

//  执行英文文本到视频文件的生成(未使用加速器加速器)
void CQtAI::Text_To_Video_NoAcc(
    QString prompt,                      //  输入的英文文本
    int steps,                           //  推理步数
    CPyAI::SInferenceArgs args,          //  推理参数
    QString out_gif_file,                //  输出的gif动图文件
    QString out_mp4_file                 //  输出的mp4视频文件
    )
{
    if(GetStatus() == EAISt_Busy) return;

    cmd_mutex.lock();
    this->cur_cmd = EAIcmd_ExTextToVideo_NoAcc;
    this->ttv_prompt = prompt;
    this->ttv_steps = steps;
    this->ttv_args = args;
    this->ttv_out_gif_file = out_gif_file;
    this->ttv_out_mp4_file = out_mp4_file;
    this->cur_st = EAISt_Busy;    //  设置忙
    cmd_mutex.unlock();

    sem_cmd.release();
}

void CQtAI::Release(void)
{
    cmd_mutex.lock();
    this->cur_cmd = EAIcmd_Release;
    cmd_mutex.unlock();

    sem_cmd.release();
}
