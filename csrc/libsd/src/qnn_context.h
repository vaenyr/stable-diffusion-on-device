#ifndef LIBSD_QNN_CONTEXT_H
#define LIBSD_QNN_CONTEXT_H

namespace libsd {

enum QnnBackend {
    HTP
};

class QnnContext {
public:
    QnnContext(std::string backend, std::list<std::string> const& extra_)
};

}

#endif // LIBSD_QNN_CONTEXT_H
