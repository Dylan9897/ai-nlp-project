import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

def send_email_with_attachment(
    subject, 
    message, 
    smtp_address,
    smtp_port,
    from_addr, 
    to_addr, 
    password, 
    filename_list
    ):
    """
    subject：标题
    message：内容
    smtp_address：SMTP服务器地址
    smtp_port：SMTP服务器端口
    from_addr：发送方
    to_addr：接收方
    password：密码
    filename_list：附件列表
    """
    # 创建MIMEMultipart对象
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject

    # 添加邮件正文
    body = MIMEText(message, 'plain')
    msg.attach(body)

    # 添加附件
    for filename in filename_list:
        with open(filename, "rb") as file:
            part = MIMEApplication(file.read(), Name=filename.split('/')[-1])
            part['Content-Disposition'] = f'attachment; filename="{filename.split("/")[-1]}"'
            msg.attach(part)

    # 邮件服务器设置
    server = smtplib.SMTP_SSL(smtp_address, smtp_port)
    print("start to login")

    server.login(from_addr, password)
    print("success for login")

    # 发送邮件
    text = msg.as_string()
    server.sendmail(from_addr, to_addr, text)

    # 关闭连接
    server.quit()

