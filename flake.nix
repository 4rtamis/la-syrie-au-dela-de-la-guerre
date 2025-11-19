{
  description = "Dev shell with atkinson-hyperlegible-next font installed";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
  in {
    devShells.${system}.default = pkgs.mkShell {
      # Packages installed and available in nix develop
      buildInputs = [
        pkgs.libertinus
      ];

      # Optionally make the font visible to fontconfig
      FONTCONFIG_FILE = pkgs.makeFontsConf {
        fontDirectories = [pkgs.atkinson-hyperlegible-next];
      };
    };
  };
}
