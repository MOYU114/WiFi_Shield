#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDateEdit>
#include <QTextEdit>
#include <QTableWidget>
#include <QPushButton>
#include <QList>
#include <QPair>
#include <QString>
#include <QFile>
#include <QTextStream>
#include <QLineEdit>


struct LogEntry {
    QString level;
    QString time;
    QString detailedTime;
    QString action;
};

class LogWindow : public QWidget {
    Q_OBJECT

public:
    LogWindow() {
        setFixedSize(700, 400); // 设置窗口大小
        setWindowTitle("日志系统");

        auto *layout = new QVBoxLayout(this);
        dateEdit = new QDateEdit(this);
        dateEdit->setCalendarPopup(true);
        dateEdit->setDate(QDate::currentDate());

        readButton = new QPushButton("读取日志", this); // 新增读取按钮

        logTable = new QTableWidget(this);
        logTable->setColumnCount(4);
        logTable->setHorizontalHeaderLabels(QStringList() << "危险等级" << "日期" << "详细时间" << "动作");
        logTable->setEditTriggers(QAbstractItemView::NoEditTriggers);

        auto *buttonLayout = new QHBoxLayout;
        buttonLayout->addWidget(dateEdit);
        buttonLayout->addWidget(readButton);

        layout->addLayout(buttonLayout);
        layout->addWidget(logTable);

        searchEdit = new QLineEdit(this); // 创建搜索框
        searchButton = new QPushButton("搜索动作", this); // 创建搜索按钮

        auto *searchLayout = new QHBoxLayout; // 创建水平布局
        searchLayout->addWidget(searchEdit);
        searchLayout->addWidget(searchButton);

        layout->addLayout(searchLayout); // 将搜索布局添加到主布局

        connect(dateEdit, &QDateEdit::dateChanged, this, &LogWindow::filterLogs);
        connect(searchButton, &QPushButton::clicked, this, &LogWindow::searchLogs);
        connect(readButton, &QPushButton::clicked, this, &LogWindow::readLogFile);

        // 从文件中读取日志
        readLogFile();
    }

private slots:
    void filterLogs() {
        QString selectedDate = dateEdit->date().toString("yyyy-MM-dd");
        logTable->setRowCount(0); // 清空表格内容

        for (auto &log : logs) {
            QString logDate = log.time; // 提取日志中的日期
            if (logDate == selectedDate) {
                int row = logTable->rowCount();
                logTable->insertRow(row);
                QTableWidgetItem *levelItem = new QTableWidgetItem(log.level);
                QTableWidgetItem *timeItem = new QTableWidgetItem(log.time);
                QTableWidgetItem *detailedTimeItem = new QTableWidgetItem(log.detailedTime);
                QTableWidgetItem *actionItem = new QTableWidgetItem(log.action);

                logTable->setItem(row, 0, levelItem);
                logTable->setItem(row, 1, timeItem);
                logTable->setItem(row, 2, detailedTimeItem);
                logTable->setItem(row, 3, actionItem);
            }
        }
        logTable->resizeColumnsToContents();
    }

    void searchLogs() {
        QString keyword = searchEdit->text().trimmed(); // 获取搜索关键词
        QString selectedDate = dateEdit->date().toString("yyyy-MM-dd");
        logTable->setRowCount(0); // 清空表格内容

        for (auto &log : logs) {
            QStringList parts = log.action.split(" ");
            if (parts[0].contains(keyword, Qt::CaseInsensitive) && log.time == selectedDate) { // 不区分大小写的搜索
                int row = logTable->rowCount();
                logTable->insertRow(row);
                logTable->setItem(row, 0, new QTableWidgetItem(log.level));
                logTable->setItem(row, 1, new QTableWidgetItem(log.time));
                logTable->setItem(row, 2, new QTableWidgetItem(log.detailedTime));
                logTable->setItem(row, 3, new QTableWidgetItem(log.action));
            }
        }
        logTable->resizeColumnsToContents();
    }

    void readLogFile() {
        logs = readLogsFromFile("E:\\college\\log_view\\log\\log\\log.txt");
        filterLogs(); // 重新筛选并显示日志
    }

private:
    QDateEdit *dateEdit;
    QPushButton *readButton;
    QTableWidget *logTable;
    QList<LogEntry> logs;
    QLineEdit *searchEdit; // 搜索框
    QPushButton *searchButton; // 搜索按钮

    QList<LogEntry> readLogsFromFile(const QString &fileName) {
        QList<LogEntry> logs;
        QFile file(fileName);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&file);
            while (!in.atEnd()) {
                QString line = in.readLine();
                QStringList parts = line.split(" - ");
                if (parts.length() >= 2) {
                    LogEntry entry;
                    entry.level = parts[0];
                    entry.time = parts[1].section(" ", 0, 0); // 提取日期和时间部分
                    entry.detailedTime = parts[1].section(" ", 1, 1);
                    entry.action = parts[1].section(" ", 2); // 提取动作部分
                    logs.append(entry);
                }
            }
            file.close();
        }
        return logs;
    }
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    LogWindow window;
    window.show();
    return app.exec();
}
