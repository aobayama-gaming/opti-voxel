#define DOCTEST_CONFIG_IMPLEMENT
// Godot owns the signal handlers; doctest must not swap them while the engine runs.
#define DOCTEST_CONFIG_NO_POSIX_SIGNALS
#include <doctest.h>

#include "unit_tests.hpp"

#include <godot_cpp/core/class_db.hpp>

namespace {

// Keeps the totals of the last run (doctest only prints them).
struct RunTotals : public doctest::IReporter {
    static doctest::TestRunStats last;

    explicit RunTotals(const doctest::ContextOptions &) {}

    void test_run_end(const doctest::TestRunStats &p_stats) override { last = p_stats; }

    void report_query(const doctest::QueryData &) override {}
    void test_run_start() override {}
    void test_case_start(const doctest::TestCaseData &) override {}
    void test_case_reenter(const doctest::TestCaseData &) override {}
    void test_case_end(const doctest::CurrentTestCaseStats &) override {}
    void test_case_exception(const doctest::TestCaseException &) override {}
    void subcase_start(const doctest::SubcaseSignature &) override {}
    void subcase_end() override {}
    void log_assert(const doctest::AssertData &) override {}
    void log_message(const doctest::MessageData &) override {}
    void test_case_skipped(const doctest::TestCaseData &) override {}
};

doctest::TestRunStats RunTotals::last;

} // namespace

DOCTEST_REGISTER_LISTENER("run_totals", 1, RunTotals);

void UnitTests::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run", "filter"), &UnitTests::run);
}

Dictionary UnitTests::run(const String &p_filter) {
    doctest::Context context;
    context.setOption("no-colors", true);
    context.setOption("no-breaks", true);
    context.setOption("no-path-filenames", true);
    if (!p_filter.is_empty()) {
        context.addFilter("test-case", p_filter.utf8().get_data());
    }
    RunTotals::last = doctest::TestRunStats();
    const int result = context.run();

    const doctest::TestRunStats &totals = RunTotals::last;
    Dictionary report;
    report["ok"] = result == 0 && totals.numTestCasesPassingFilters > 0;
    report["test_cases"] = (int64_t)totals.numTestCasesPassingFilters;
    report["test_cases_failed"] = (int64_t)totals.numTestCasesFailed;
    report["asserts"] = (int64_t)totals.numAsserts;
    report["asserts_failed"] = (int64_t)totals.numAssertsFailed;
    return report;
}
