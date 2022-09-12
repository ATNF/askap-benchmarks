/*! \file time_util.cpp
 *  \brief Get timing
 */

#include "profile_util.h"

/// get the time taken to do some comptue 
namespace profiling_util {

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

} 

