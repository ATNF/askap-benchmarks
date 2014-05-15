// Include own header file first
#include "Stopwatch.h"

// System includes
#include <unistd.h>
#include <sys/times.h>
#include <stdexcept>

Stopwatch::Stopwatch() : m_start(-1)
{
}

Stopwatch::~Stopwatch()
{
}

void Stopwatch::start()
{
	struct tms t;
	m_start = times(&t);
	if (m_start == -1)
	{
		throw std::runtime_error("Error calling times()");
	}
}

double Stopwatch::stop()
{
	struct tms t;
	clock_t stop = times(&t);

	if (m_start == -1)
	{
		throw std::runtime_error("Start time not set");
	}

	if (stop == -1)
	{
		throw std::runtime_error("Error calling times()");
	}

	return (static_cast<double>(stop - m_start)) / (static_cast<double>(sysconf(_SC_CLK_TCK)));
}
