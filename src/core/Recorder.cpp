// =============================================================================
// Recorder.cpp -- binary / CSV (de)serialization for recorded episodes.
// =============================================================================
#include "core/Recorder.hpp"
#include "util/Logger.hpp"

#include <fstream>
#include <cstring>

namespace dp {

namespace {
// 8-byte magic so a stray/old file is rejected rather than misread.
constexpr char kMagic[8] = {'D','P','R','E','C','\0','\0','1'};
} // namespace

bool EpisodeRecorder::saveBinary(const std::string& path) const {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) { DP_LOG_ERROR("Recorder: cannot write '%s'", path.c_str()); return false; }
    const std::uint64_t count = frames_.size();
    os.write(kMagic, sizeof(kMagic));
    os.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (count > 0)
        os.write(reinterpret_cast<const char*>(frames_.data()),
                 static_cast<std::streamsize>(count * sizeof(RecordFrame)));
    return static_cast<bool>(os);
}

bool EpisodeRecorder::loadBinary(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { DP_LOG_ERROR("Recorder: cannot read '%s'", path.c_str()); return false; }
    char magic[8];
    is.read(magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        DP_LOG_ERROR("Recorder: bad magic in '%s'", path.c_str());
        return false;
    }
    std::uint64_t count = 0;
    is.read(reinterpret_cast<char*>(&count), sizeof(count));
    frames_.resize(count);
    if (count > 0)
        is.read(reinterpret_cast<char*>(frames_.data()),
                static_cast<std::streamsize>(count * sizeof(RecordFrame)));
    return static_cast<bool>(is);
}

bool EpisodeRecorder::saveCsv(const std::string& path) const {
    std::ofstream os(path, std::ios::trunc);
    if (!os) { DP_LOG_ERROR("Recorder: cannot write '%s'", path.c_str()); return false; }
    os << "frame,theta1,theta2,omega1,omega2,torque1,torque2,reward,value,meanSigma,energy\n";
    for (std::size_t i = 0; i < frames_.size(); ++i) {
        const RecordFrame& f = frames_[i];
        os << i << ',' << f.theta1 << ',' << f.theta2 << ',' << f.omega1 << ','
           << f.omega2 << ',' << f.torque1 << ',' << f.torque2 << ',' << f.reward
           << ',' << f.value << ',' << f.meanSigma << ',' << f.energy << '\n';
    }
    return static_cast<bool>(os);
}

} // namespace dp
