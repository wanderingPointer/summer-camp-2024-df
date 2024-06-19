#include <barvinok/barvinok.h>
#include <isl/val.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main(int argv, char **argc) {
  if (argv < 4) {
    printf("Memory Estimator\nUsage: memEst <ifname> <ofname> <n>\n");
    exit(1);
  }
  auto *ifname = argc[1];
  auto *ofname = argc[2];
  int n = atoi(argc[3]);

  ifstream fin{ifname};
  ofstream fout{ofname};

  string entryDomainStr;
  string tensorDomainStr;
  string layoutStr;

  auto ctx = isl_ctx_alloc();
  getline(fin, entryDomainStr);
  auto entryDomain = isl_union_set_read_from_str(ctx, entryDomainStr.c_str());

  for (int i = 0; i < n; ++i) {
    getline(fin, tensorDomainStr);
    getline(fin, layoutStr);

    auto tensorDomain =
        isl_union_set_read_from_str(ctx, tensorDomainStr.c_str());
    auto layout = isl_union_map_read_from_str(ctx, layoutStr.c_str());
    auto tensorAccess =
        isl_union_set_apply(isl_union_set_copy(entryDomain), layout);
    auto intersect = isl_union_set_intersect(tensorAccess, tensorDomain);
    auto size = isl_union_set_card(intersect);

    isl_printer *p = isl_printer_to_str(isl_union_pw_qpolynomial_get_ctx(size));
    p = isl_printer_set_output_format(p, ISL_FORMAT_ISL);
    p = isl_printer_print_union_pw_qpolynomial(p, size);
    char *s = isl_printer_get_str(p);
    int ret = atoi(s + 1);
    isl_union_pw_qpolynomial_free(size);
    isl_printer_free(p);
    fout << ret << endl;
  }

  isl_union_set_free(entryDomain);
  isl_ctx_free(ctx);

  return 0;
}