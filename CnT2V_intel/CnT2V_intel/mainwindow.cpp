#include "mainwindow.h"
#include "ui_mainwindow.h"

//  全局变量，用于步骤更新
extern int g_steps;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //  初始化UI
    QImage img_out(ui->label_outputvideo->width(), ui->label_outputvideo->height(), QImage::Format_RGB888);
    img_out.fill(QColor(0, 0, 255));
    QPainter* painter = new QPainter();
    painter->begin(&img_out);
    painter->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);    //  抗锯齿
    painter->setPen(QPen(QColor(255,255,255), 5, Qt::SolidLine));
    painter->drawText(QPoint(ui->label_outputvideo->width() / 3, ui->label_outputvideo->height() / 2), "个人赛道-王祥福");
    painter->end();
    ui->label_outputvideo->setPixmap(QPixmap::fromImage(img_out));
    delete  painter;

    //  设置当前忙状态
    is_cur_busy = true;

    //  添加单选按钮到加速器模式单选组
    acc_mode_grp.addButton(ui->radioButton_gpunup, EAcceleratorMode_iGPU_NPU);
    acc_mode_grp.addButton(ui->radioButton_noacc,  EAcceleratorMode_NoAcc);

    //  设置默认加速器模式
    cur_acc_mode = EAcceleratorMode_iGPU_NPU;
    ui->radioButton_gpunup->setChecked(true);
    ui->radioButton_noacc ->setChecked(false);

    //  连接加速器模式选择
    connect(&acc_mode_grp, SIGNAL(buttonClicked(int)), this, SLOT(slot_AccModeSelect(int)));

    //  提示
    ui->progressBar->setFormat("Initializing environment, please wait...");

    //  关闭控件
    this->Disable_UI();

    //  设置视频时长默认值
    ui->spinBox_vd->setValue(2);

    //  视频播放
    m_mediaplayer = nullptr;

    //  视频播放状态
    video_st = EVideoSt_Stop;

    //  视频探针
    m_videoprobe = nullptr;

    //  初始化标志
    only_export_IR = true;
    only_tsl_flag  = true;

    //  初始化上一次的进度值
    last_steps = 0;

    //  初始化AI环境
    connect(&qtai, &CQtAI::send_translate_cn2en_finish, this, &MainWindow::slot_OnTranslateCn2EnFinish);
    connect(&qtai, &CQtAI::send_exportIR_finish,        this, &MainWindow::slot_OnExportIRFinish);
    connect(&qtai, &CQtAI::send_text_to_video_finish,   this, &MainWindow::slot_OnTextToVideoFinish);
    connect(&qtai, &CQtAI::send_environment_ready,      this, &MainWindow::slot_OnAIEnvReady);
    qtai.Init();
    qtai.start();

    //  定时器初始化
    timer = new QTimer(this);
    timer->setInterval(30);
    connect(timer, SIGNAL(timeout()), this, SLOT(slot_timeout()));
    timer->start();
}

MainWindow::~MainWindow()
{
    //  释放视频资源
    if(m_mediaplayer != nullptr)
    {
        m_mediaplayer->stop();
        delete m_mediaplayer;
        m_mediaplayer = nullptr;
    }

    //  释放视频探针
    if(m_videoprobe != nullptr)
    {
        delete m_videoprobe;
        m_videoprobe = nullptr;
    }

    //  释放定时器
    if(timer != nullptr)
    {
        timer->stop();
        delete timer;
        timer = nullptr;
    }

    //  释放AI资源
    qtai.Release();

    //  释放UI资源
    delete ui;
}

//  关闭UI
void MainWindow::Disable_UI(void)
{
    ui->pushButton_play          ->setEnabled(false);
    ui->pushButton_pause         ->setEnabled(false);
    ui->pushButton_stop          ->setEnabled(false);
    ui->pushButton_bycn          ->setEnabled(false);
    ui->pushButton_byen          ->setEnabled(false);
    ui->pushButton_tslonly       ->setEnabled(false);
    ui->pushButton_export_compile->setEnabled(false);
    ui->spinBox_vd               ->setEnabled(false);
    ui->spinBox_step             ->setEnabled(false);
    ui->spinBox_res_w            ->setEnabled(false);
    ui->spinBox_res_h            ->setEnabled(false);
    ui->radioButton_gpunup       ->setEnabled(false);
    ui->radioButton_noacc        ->setEnabled(false);
}

//  打开UI
void MainWindow::Enable_UI(void)
{
    ui->pushButton_play          ->setEnabled(true);
    ui->pushButton_pause         ->setEnabled(true);
    ui->pushButton_stop          ->setEnabled(true);
    ui->pushButton_bycn          ->setEnabled(true);
    ui->pushButton_byen          ->setEnabled(true);
    ui->pushButton_tslonly       ->setEnabled(true);
    ui->pushButton_export_compile->setEnabled(true);
    ui->spinBox_vd               ->setEnabled(true);
    ui->spinBox_step             ->setEnabled(true);
    ui->spinBox_res_w            ->setEnabled(true);
    ui->spinBox_res_h            ->setEnabled(true);
    ui->radioButton_gpunup       ->setEnabled(true);
    ui->radioButton_noacc        ->setEnabled(true);
}

//  加速器模式单选槽
void MainWindow::slot_AccModeSelect(int mode)
{
    cur_acc_mode = static_cast<EAcceleratorMode>(mode);

    if(cur_acc_mode == EAcceleratorMode_iGPU_NPU) qDebug("now set iGPU+NPU mode");
    else                                          qDebug("now set No Acc mode");
}

//  当AI环境就绪
void MainWindow::slot_OnAIEnvReady(void)
{
    //  尝试获取本地已有参数
    CPyAI::SInferenceArgs args;
    int re = qtai.GetLocalIRargs(&args);

    //  当参数可用，覆盖UI默认参数
    if(re == 0)
    {
        //  将参数更新到UI界面
        ui->spinBox_res_w->setValue(args.width);
        ui->spinBox_res_h->setValue(args.height);
        ui->spinBox_vd   ->setValue(args.total_frames / 8);
    }

    //  开启UI
    this->Enable_UI();

    //  提示
    is_cur_busy = false;
    ui->progressBar->setFormat("Ready!!");
}

//  当完成一次翻译
void MainWindow::slot_OnTranslateCn2EnFinish(QString out_text, qint64 run_time_ns)
{
    //  检查文字内容的最后1个字符
    if(!out_text.isEmpty())
    {
        //  当为英文句号的时候，删除该字符
        if(*(out_text.end()-1) == '.')
        {
            out_text.remove(out_text.length() - 1, 1);
        }
    }

    //  更新输出文字内容
    ui->textEdit_en->setText(out_text);
    std::string tmp_str = out_text.toStdString();
    qDebug("out_text = [%s]", tmp_str.c_str());

    //  当为只进行翻译操作
    if(only_tsl_flag)
    {
        //  忙完
        is_cur_busy = false;

        //  开启UI
        this->Enable_UI();

        //  设置状态
        //  当为正常的文字内容
        if(!out_text.isEmpty())
        {
            long long ms = run_time_ns / 1000000LL;
            int sec = static_cast<int>(ms / 1000);
            int min = sec / 60;
            int hour = min / 60;
            QString str = QString::asprintf("Translate Done(%d:%02d:%02d.%03lld)",
                                            hour, min % 60, sec % 60, ms % 1000LL
                                           );
            ui->progressBar->setFormat(str);
        }
        //  当输出文字内容为空
        else
        {
            //  提示错误
            ui->progressBar->setFormat("Translation result is empty!!");
        }
    }
    //  否则直接执行文字生成视频
    else
    {
        //  当为正常的文字内容
        if(!out_text.isEmpty())
        {
            //  删除缓存文件
        #if BEGIN_DELETE_VIDEO_TMP_FILE
            std::string cmd;
            cmd = "rm -rf ";
            cmd += TEXT_TO_VIDEO_GIF_TMP_FILE;
            system(cmd.c_str());
            cmd = "rm -rf ";
            cmd += TEXT_TO_VIDEO_MP4_TMP_FILE;
            system(cmd.c_str());
        #endif  //  BEGIN_DELETE_VIDEO_TMP_FILE

            //  等价于英文生成视频按钮被单击
            this->on_pushButton_byen_clicked();
        }
        //  当输出文字内容为空
        else
        {
            //  忙完
            is_cur_busy = false;

            //  开启UI
            this->Enable_UI();

            //  提示错误
            ui->progressBar->setFormat("Translation result is empty!!");
        }
    }
}

//  当完成一次IR模型导出和编译
void MainWindow::slot_OnExportIRFinish(qint64 run_time_ns)
{
    //  当本次只是进行模型导出
    if(only_export_IR)
    {
        //  忙完
        is_cur_busy = false;

        //  开启UI
        this->Enable_UI();

        //  提示信息
        long long ms = run_time_ns / 1000000LL;
        int sec = static_cast<int>(ms / 1000);
        int min = sec / 60;
        int hour = min / 60;
        QString str = QString::asprintf("Export IR&Compile Done(%d:%02d:%02d.%03lld)",
                                        hour, min % 60, sec % 60, ms % 1000LL
                                       );
        ui->progressBar->setFormat(str);
    }
    //  否则，继续进行文字生成视频的推理
    else
    {
        //  当为使用iGPU+NPU加速器
        if(cur_acc_mode == EAcceleratorMode_iGPU_NPU)
        {
            //  检查输入文字
            QString in_text = ui->textEdit_en->toPlainText();
            if(in_text.isEmpty())
            {
                //  提示
                ui->progressBar->setFormat("Input English Text is NULL!");
                this->Enable_UI();
                return;
            }

            //  调用加速器的文字生成视频
            qtai.Text_To_Video_with_iGPU_NPU(
                     in_text,
                     ui->spinBox_step->value(),
                     TEXT_TO_VIDEO_GIF_TMP_FILE,
                     TEXT_TO_VIDEO_MP4_TMP_FILE
                    );

            //  设置状态
            progress_suffix_str = "(iGPU+NPU)...";
            ui->progressBar->setFormat("Generateing Video(iGPU+NPU)...");
        }
        //  不使用加速器
        else
        {
            //  这种情况不可能发生
            //  设置状态
            ui->progressBar->setFormat("unknown error");
            this->Enable_UI();
        }
    }
}

//  当完成一次文字生成视频
void MainWindow::slot_OnTextToVideoFinish(qint64 run_time_ns, bool with_iGPU_NPU)
{
    //  忙完
    is_cur_busy = false;

    //  开启UI
    this->Enable_UI();

    //  设置状态
    long long ms = run_time_ns / 1000000LL;
    int sec = static_cast<int>(ms / 1000);
    int min = sec / 60;
    int hour = min / 60;
    QString str = QString::asprintf("Finish (%d:%02d:%02d.%03lld) with %s",
                                    hour, min % 60, sec % 60, ms % 1000LL,
                                    (with_iGPU_NPU)?"iGPU+NPU":"No Acc"
                                   );
    ui->progressBar->setFormat(str);

    //  载入视频
    QString currentPath = QCoreApplication::applicationDirPath();
    this->LoadVideo(currentPath + "/" + TEXT_TO_VIDEO_MP4_TMP_FILE);

    //  自动播放视频
    this->on_pushButton_play_clicked();
}

//  定时器
void MainWindow::slot_timeout()
{
    //  进度变量
    static int cur_proc = 100;

    //  当忙
    if(is_cur_busy)
    {
        if(cur_proc >= 100) cur_proc = 0;
        else                cur_proc = cur_proc + 1;
    }
    //  当忙完
    else
    {
        cur_proc = 100;
    }

    //  更新到控件
    ui->progressBar->setValue(cur_proc);

    //  当进度值变化
    if(last_steps != g_steps)
    {
        //  获取当前进度
        int now_steps = g_steps;

        //  更新进度
        QString tmp_str;
        tmp_str = QString::asprintf("%d%% Step %d of %d ",
                         now_steps * 100 / ui->spinBox_step->value(),
                         now_steps, ui->spinBox_step->value());
        tmp_str += progress_suffix_str;
        ui->progressBar->setFormat(tmp_str);

        //  更新
        last_steps = now_steps;
    }
}

//  载入视频
void MainWindow::LoadVideo(QString filename)
{
    //  释放之前的资源
    if(m_mediaplayer != nullptr)
    {
        m_mediaplayer->stop();
        delete m_mediaplayer;
        m_mediaplayer = nullptr;
    }

    //  创建视频播放
    m_mediaplayer = new QMediaPlayer(this);

    //  检查视频探针
    if(m_videoprobe == nullptr)
    {
        //  创建视频帧探针
        m_videoprobe = new QVideoProbe;

        //  链接视频帧探针的信号
        connect(m_videoprobe, SIGNAL(videoFrameProbed(QVideoFrame)), this, SLOT(slot_OnVideoProbeFrame(QVideoFrame)));
    }

    //  配置探针的源数据
    m_videoprobe ->setSource(m_mediaplayer);

    //  载入视频文件
    m_mediaplayer->setMedia(QUrl::fromLocalFile(filename));

    //  载入后，先暂停
    m_mediaplayer->pause();
    video_st = EVideoSt_Pause;
}

//  视频流探针槽
void MainWindow::slot_OnVideoProbeFrame(const QVideoFrame& frame)
{
    //  检查
    if(!frame.isValid()) return;

    //  复制一份当前帧的副本
    QVideoFrame cloneFrame(frame);
    cloneFrame.map(QAbstractVideoBuffer::ReadOnly);   //  采用只读方式映射

    //  构造输出帧图像
    QImage img(cloneFrame.width(), cloneFrame.height(), QImage::Format_RGB888);
    unsigned char* in_buf = cloneFrame.bits();        //  视频流原始数据帧缓冲区
    unsigned char* out_buf = img.bits();              //  视频流输出数据帧缓冲区

    //  根据源帧的格式进行转换
    //  这里只处理常见格式
    if(cloneFrame.pixelFormat()==QVideoFrame::Format_YV12)
    {
        int y=0;
        int x=0;
        for(y=0;y<cloneFrame.height();y++)
        {
            for(x=0;x<cloneFrame.width();x++)
            {
                unsigned char Y = static_cast<unsigned char>(in_buf[y*cloneFrame.width() + x]);
                unsigned char V = static_cast<unsigned char>(in_buf[cloneFrame.width() * cloneFrame.height() + (y/2)*(cloneFrame.width()/2) + x/2]);
                unsigned char U = static_cast<unsigned char>(in_buf[cloneFrame.width() * cloneFrame.height() + ((cloneFrame.width()/2) * (cloneFrame.height()/2)) + (y/2)*(cloneFrame.width()/2) + x/2]);

                double R = Y + 1.402 * (V-128);
                double G = Y - 0.344136 * (U-128) - 0.714136 * (V-128);
                double B = Y + 1.772 * (U-128);

                if(R >= 255) R = 255;
                if(R <= 0) R = 0;
                if(G >= 255) G = 255;
                if(G <= 0) G = 0;
                if(B >= 255) B = 255;
                if(B <= 0) B = 0;

                out_buf[y*img.bytesPerLine() + x*3 + 0] = static_cast<unsigned char>(R);
                out_buf[y*img.bytesPerLine() + x*3 + 1] = static_cast<unsigned char>(G);
                out_buf[y*img.bytesPerLine() + x*3 + 2] = static_cast<unsigned char>(B);
            }
        }
    }
    else if(cloneFrame.pixelFormat()==QVideoFrame::Format_YUV420P)
    {
        int y=0;
        int x=0;
        for(y=0;y<cloneFrame.height();y++)
        {
            for(x=0;x<cloneFrame.width();x++)
            {
                unsigned char Y = static_cast<unsigned char>(in_buf[y*cloneFrame.width() + x]);
                unsigned char U = static_cast<unsigned char>(in_buf[cloneFrame.width() * cloneFrame.height() + (y/2)*(cloneFrame.width()/2) + x/2]);
                unsigned char V = static_cast<unsigned char>(in_buf[cloneFrame.width() * cloneFrame.height() + ((cloneFrame.width()/2) * (cloneFrame.height()/2)) + (y/2)*(cloneFrame.width()/2) + x/2]);

                double R = Y + 1.402 * (V-128);
                double G = Y - 0.344136 * (U-128) - 0.714136 * (V-128);
                double B = Y + 1.772 * (U-128);

                if(R >= 255) R = 255;
                if(R <= 0) R = 0;
                if(G >= 255) G = 255;
                if(G <= 0) G = 0;
                if(B >= 255) B = 255;
                if(B <= 0) B = 0;

                out_buf[y*img.bytesPerLine() + x*3 + 0] = static_cast<unsigned char>(R);
                out_buf[y*img.bytesPerLine() + x*3 + 1] = static_cast<unsigned char>(G);
                out_buf[y*img.bytesPerLine() + x*3 + 2] = static_cast<unsigned char>(B);
            }
        }
    }
    else if(cloneFrame.pixelFormat()==QVideoFrame::Format_NV12)
    {
        int y=0;
        int x=0;
        for(y=0;y<cloneFrame.height();y++)
        {
            for(x=0;x<cloneFrame.width();x++)
            {
                unsigned char Y = static_cast<unsigned char>(in_buf[y*cloneFrame.bytesPerLine() + x]);
                unsigned char U = static_cast<unsigned char>(in_buf[cloneFrame.bytesPerLine() * cloneFrame.height() + (y/2)*(cloneFrame.bytesPerLine()) + (x&(~1)) + 0]);
                unsigned char V = static_cast<unsigned char>(in_buf[cloneFrame.bytesPerLine() * cloneFrame.height() + (y/2)*(cloneFrame.bytesPerLine()) + (x&(~1)) + 1]);

                double R = Y + 1.402 * (V-128);
                double G = Y - 0.344136 * (U-128) - 0.714136 * (V-128);
                double B = Y + 1.772 * (U-128);

                if(R >= 255) R = 255;
                if(R <= 0) R = 0;
                if(G >= 255) G = 255;
                if(G <= 0) G = 0;
                if(B >= 255) B = 255;
                if(B <= 0) B = 0;

                out_buf[y*img.bytesPerLine() + x*3 + 0] = static_cast<unsigned char>(R);
                out_buf[y*img.bytesPerLine() + x*3 + 1] = static_cast<unsigned char>(G);
                out_buf[y*img.bytesPerLine() + x*3 + 2] = static_cast<unsigned char>(B);
            }
        }
    }
    else if(cloneFrame.pixelFormat()==QVideoFrame::Format_RGB32)
    {
        int y=0;
        int x=0;
        for(y=0;y<cloneFrame.height();y++)
        {
            for(x=0;x<cloneFrame.width();x++)
            {
                unsigned char R = static_cast<unsigned char>(in_buf[y*cloneFrame.bytesPerLine() + x*4 + 2]);
                unsigned char G = static_cast<unsigned char>(in_buf[y*cloneFrame.bytesPerLine() + x*4 + 1]);
                unsigned char B = static_cast<unsigned char>(in_buf[y*cloneFrame.bytesPerLine() + x*4 + 0]);

                out_buf[y*img.bytesPerLine() + x*3 + 0] = static_cast<unsigned char>(R);
                out_buf[y*img.bytesPerLine() + x*3 + 1] = static_cast<unsigned char>(G);
                out_buf[y*img.bytesPerLine() + x*3 + 2] = static_cast<unsigned char>(B);
            }
        }
    }
    else
    {
        qDebug("Unsupport in_fmt = %d w=%d, h=%d, bl=%d", cloneFrame.pixelFormat(), cloneFrame.width(), cloneFrame.height(), cloneFrame.bytesPerLine());
    }

    //  根据输出控件的尺寸进行缩放
    img = img.scaled(ui->label_outputvideo->width(), ui->label_outputvideo->height(), Qt::KeepAspectRatio);

    //  显示到控件上
    ui->label_outputvideo->setPixmap(QPixmap::fromImage(img));

    //  取消映射
    cloneFrame.unmap();
}

//  单击只将中文翻译成英文的按钮
void MainWindow::on_pushButton_tslonly_clicked()
{
    //  检查输入文字
    QString in_text = ui->textEdit_cn->toPlainText();
    if(in_text.isEmpty())
    {
        //  提示
        ui->progressBar->setFormat("Input Chinese Text is NULL!");
        return;
    }

    //  执行
    is_cur_busy = true;
    qtai.ExTranslateCn2En(in_text);

    //  设置状态
    ui->progressBar->setFormat("Translate Processing...");

    //  关闭控件
    this->Disable_UI();

    //  设置标志
    only_tsl_flag = true;
}

//  单击通过英文生成视频按钮
void MainWindow::on_pushButton_byen_clicked()
{
    //  检查输入文字
    QString in_text = ui->textEdit_en->toPlainText();
    if(in_text.isEmpty())
    {
        //  提示
        ui->progressBar->setFormat("Input English Text is NULL!");
        return;
    }

    //  获取总视频时长，并计算总帧数
    int total_frames = ui->spinBox_vd->value() * 8;

    //  检查总帧数
    if(total_frames < TOTAL_FRAME_NUM_MIN)
    {
        total_frames = TOTAL_FRAME_NUM_MIN;
    }

    //  删除临时文件
    is_cur_busy = true;
#if BEGIN_DELETE_VIDEO_TMP_FILE
    std::string cmd;
    cmd = "rm -rf ";
    cmd += TEXT_TO_VIDEO_GIF_TMP_FILE;
    system(cmd.c_str());
    cmd = "rm -rf ";
    cmd += TEXT_TO_VIDEO_MP4_TMP_FILE;
    system(cmd.c_str());
#endif  //  BEGIN_DELETE_VIDEO_TMP_FILE

    //  当为使用iGPU+NPU加速器
    if(cur_acc_mode == EAcceleratorMode_iGPU_NPU)
    {
        //  获取当前IR模型参数
        CPyAI::SInferenceArgs re_args;
        int re = qtai.GetLocalIRargs(&re_args);

        //  当获取失败,直接导出模型
        if(re != 0)
        {
            //  提示
            qDebug("No local model found, new export");

            //  构造模型参数
            CPyAI::SInferenceArgs args;
            args.args_valid   = true;
            args.width        = ui->spinBox_res_w->value();
            args.height       = ui->spinBox_res_h->value();
            args.total_frames = total_frames;

            //  设置状态
            ui->progressBar->setFormat("Export IR model & Compile model...");

            //  设置 不仅导出IR模型
            only_export_IR = false;

            //  导出IR模型
            qtai.ExConvIR(args);
        }
        //  获取成功
        else
        {
            //  检查已有模型参数和目前UI设置的参数是否一致
            //  当一致
            if((re_args.width        == ui->spinBox_res_w->value()) &&
               (re_args.height       == ui->spinBox_res_h->value()) &&
               (re_args.total_frames == total_frames)
              )
            {
                //  当编译后的IR模型可用
                if(qtai.IR_model_is_valid())
                {
                    //  提示
                    qDebug("Detected that the local model parameters have not changed, using the local model directly");

                    //  调用加速器的文字生成视频
                    qtai.Text_To_Video_with_iGPU_NPU(
                             in_text,
                             ui->spinBox_step->value(),
                             TEXT_TO_VIDEO_GIF_TMP_FILE,
                             TEXT_TO_VIDEO_MP4_TMP_FILE
                            );

                    //  设置状态
                    progress_suffix_str = "(iGPU+NPU)...";
                    ui->progressBar->setFormat("Generateing Video(iGPU+NPU)...");
                }
                //  不可用
                else
                {
                    //  提示
                    qDebug("Local model not compiled, proceed with model compilation");

                    //  构造模型参数
                    CPyAI::SInferenceArgs args;
                    args.args_valid   = true;
                    args.width        = ui->spinBox_res_w->value();
                    args.height       = ui->spinBox_res_h->value();
                    args.total_frames = total_frames;

                    //  设置状态
                    ui->progressBar->setFormat("Compile model...");

                    //  设置 不仅导出IR模型
                    only_export_IR = false;

                    //  导出IR模型
                    qtai.ExConvIR(args);
                }

            }
            //  不一致,删除老的，重新导出
            else
            {
                //  提示
                qDebug("Local model parameters changed, re export model");

                //  删除已导出IR模型
                qtai.DeleteLocalIRmodel();

                //  构造模型参数
                CPyAI::SInferenceArgs args;
                args.args_valid   = true;
                args.width        = ui->spinBox_res_w->value();
                args.height       = ui->spinBox_res_h->value();
                args.total_frames = total_frames;

                //  设置状态
                ui->progressBar->setFormat("Export IR model & Compile model...");

                //  设置 不仅导出IR模型
                only_export_IR = false;

                //  导出IR模型
                qtai.ExConvIR(args);
            }
        }
    }
    //  不使用加速器
    else
    {
        //  构造模型参数
        CPyAI::SInferenceArgs args;
        args.args_valid   = true;
        args.width        = ui->spinBox_res_w->value();
        args.height       = ui->spinBox_res_h->value();
        args.total_frames = total_frames;

        //  调用非加速器的文字生成视频
        qtai.Text_To_Video_NoAcc(
                 in_text,
                 ui->spinBox_step->value(),
                 args,
                 TEXT_TO_VIDEO_GIF_TMP_FILE,
                 TEXT_TO_VIDEO_MP4_TMP_FILE
                );

        //  设置状态
        progress_suffix_str = "(No Acc)...";
        ui->progressBar->setFormat("Generateing Video(No Acc)...");
    }

    //  关闭控件
    this->Disable_UI();
}

//  当单击只导出IR模型
void MainWindow::on_pushButton_export_compile_clicked()
{
    //  获取总视频时长，并计算总帧数
    int total_frames = ui->spinBox_vd->value() * 8;

    //  检查总帧数
    if(total_frames < TOTAL_FRAME_NUM_MIN)
    {
        total_frames = TOTAL_FRAME_NUM_MIN;
    }

    //  构造模型参数
    CPyAI::SInferenceArgs args;
    args.args_valid   = true;
    args.width        = ui->spinBox_res_w->value();
    args.height       = ui->spinBox_res_h->value();
    args.total_frames = total_frames;

    //  设置状态
    ui->progressBar->setFormat("Export IR model & Compile model...");
    is_cur_busy = true;
    this->Disable_UI();

    //  设置 仅导出IR模型
    only_export_IR = true;

    //  导出IR模型
    qtai.ExConvIR(args);
}

//  单击视频播放按钮
void MainWindow::on_pushButton_play_clicked()
{
    //  播放
    m_mediaplayer->play();
    video_st = EVideoSt_Play;
}

//  单击视频暂停按钮
void MainWindow::on_pushButton_pause_clicked()
{
    //  暂停
    m_mediaplayer->pause();
    video_st = EVideoSt_Pause;
}

//  单击视频停止按钮
void MainWindow::on_pushButton_stop_clicked()
{
    //  停止
    m_mediaplayer->stop();
    m_mediaplayer->pause();
    video_st = EVideoSt_Stop;
}

//  单击通过中文生成视频按钮
void MainWindow::on_pushButton_bycn_clicked()
{
    //  检查输入文字
    QString in_text = ui->textEdit_cn->toPlainText();
    if(in_text.isEmpty())
    {
        //  提示
        ui->progressBar->setFormat("Input Chinese Text is NULL!");
        return;
    }

    //  设置状态
    ui->progressBar->setFormat("Translate Processing...");

    //  关闭控件
    this->Disable_UI();

    //  设置标志
    only_tsl_flag = false;

    //  执行
    is_cur_busy = true;
    qtai.ExTranslateCn2En(in_text);

}

//  当视频宽度数值发生变化
void MainWindow::on_spinBox_res_w_valueChanged(int arg1)
{
    //  当数值小于最小值
    if(arg1 < 240) ui->spinBox_res_w->setValue(240);

    //  当数值大于最大值
    if(arg1 > 992) ui->spinBox_res_w->setValue(992);

    //  当不是8的倍数
    if((arg1 % 8) != 0)
    {
        volatile int mul = arg1 / 8;
        ui->spinBox_res_w->setValue(mul * 8);
    }
}

//  当视频高度数值发生变化
void MainWindow::on_spinBox_res_h_valueChanged(int arg1)
{
    //  当数值小于最小值
    if(arg1 < 240) ui->spinBox_res_h->setValue(240);

    //  当数值大于最大值
    if(arg1 > 992) ui->spinBox_res_h->setValue(992);

    //  当不是8的倍数
    if((arg1 % 8) != 0)
    {
        volatile int mul = arg1 / 8;
        ui->spinBox_res_h->setValue(mul * 8);
    }
}
