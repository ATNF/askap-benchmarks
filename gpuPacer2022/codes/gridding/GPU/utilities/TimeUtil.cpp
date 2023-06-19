/*! \file TimeUtil.cpp
 *  \brief Get timing
 */

#include "LoggerUtil.h"

/// get the time taken to do some comptue 
namespace logger_util {

    std::string __when(){
        auto log_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::string whenbuff=std::ctime(&log_time);
        whenbuff.erase(std::find(whenbuff.begin(), whenbuff.end(), '\n'), whenbuff.end());
        return whenbuff;
    }

    template <typename T>
    inline
    std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, const Timer &t) {
        os << us_time(t.get());
        return os;
    }

    std::string ReportTimeTaken(
        const Timer &t, 
        const std::string &function, 
        const std::string &line_num)
    {
        std::string new_ref = "@"+function+" L"+line_num;
        std::ostringstream report;
        report <<"Time taken between : " << new_ref << " - " << t.get_ref() << " : " << us_time(t.get());
        return report.str();
    }

    float GetTimeTaken(
        const Timer &t, 
        const std::string &function, 
        const std::string &line_num)
    {
        return static_cast<float>((t.get()));
    }

#if defined(_GPU)
    std::string ReportTimeTakenOnDevice(
        const Timer &t, 
        const std::string &function, 
        const std::string &line_num)
    {
        std::string new_ref = "@"+function+" L"+line_num;
        std::ostringstream report;
        if (t.get_use_device()) {
            report << "Time taken on device between : " << new_ref << " - " << t.get_ref() << " : " << us_time(t.get_on_device());
        }
        else {
            report << "NO DEVICE to measure : " << new_ref << " - " << t.get_ref() << " : " << us_time(t.get_on_device());
        }
        return report.str();
    }

    float GetTimeTakenOnDevice(
        const Timer &t, 
        const std::string &function, 
        const std::string &line_num)
    {
        return t.get_on_device();
    }
#endif

} 

