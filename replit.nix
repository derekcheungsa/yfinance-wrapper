{pkgs}: {
  deps = [
    pkgs.gcc
    pkgs.glibcLocales
    pkgs.postgresql
    pkgs.openssl
  ];
}
