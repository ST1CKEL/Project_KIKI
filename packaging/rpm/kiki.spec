%global kiki_python_sitelib %(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib", scheme="rpm_prefix"))')
%global kiki_python_version %(python3 -c 'import sysconfig; print(sysconfig.get_config_var("py_version_short"))')
%global debug_package %{nil}

Name:           kiki
Version:        0.8.0
Release:        5%{?dist}
Summary:        Friendly 2D AI desktop pet for Fedora Linux

License:        MIT
Source0:        %{name}-%{version}.tar.gz

BuildArch:      x86_64
BuildRequires:  appstream
BuildRequires:  desktop-file-utils
BuildRequires:  espeak-ng
BuildRequires:  gdk-pixbuf2
BuildRequires:  git-core
BuildRequires:  python3
BuildRequires:  python3-cairo
BuildRequires:  python3-cffi
BuildRequires:  python3-gobject
BuildRequires:  python3-httpx
BuildRequires:  python3-pillow
BuildRequires:  python3-pytest
BuildRequires:  systemd-rpm-macros

Requires:       espeak-ng
Requires:       git-core
Requires:       gstreamer1
Requires:       gstreamer1-plugins-base
Requires:       gstreamer1-plugins-good
Requires:       gtk4
Requires:       hicolor-icon-theme
Requires:       libadwaita
Requires:       libsecret
Requires:       python(abi) = %{kiki_python_version}
Requires:       python3 >= 3.13
Requires:       python3-cairo
Requires:       python3-cffi
Requires:       python3-gobject
Requires:       python3dist(httpx) >= 0.27
Requires:       python3dist(pillow) >= 10
Requires:       pipewire-pulseaudio
Requires:       vosk-api-devel >= 0.3.50
Requires:       xdg-terminal-exec
Requires:       xdg-utils
Recommends:     gnome-keyring
Recommends:     pipewire-utils
Recommends:     xdg-desktop-portal
Suggests:       gnome-text-editor
Suggests:       ollama
Suggests:       ptyxis
Suggests:       python3-pytest
Suggests:       spectacle
Suggests:       xdg-desktop-portal-gnome

%description
KIKI is a friendly GTK4/libadwaita desktop pet for Fedora. It can chat with
local Ollama models, show animated character states, and orchestrate a coding
agent only inside explicitly registered Git workspaces. Its desktop control
center exposes only declared, individually confirmed and audited actions.

%prep
%autosetup -n %{name}-%{version}

%build
# KIKI is pure Python. The Fedora package installs the source directly so the
# RPM can be built offline using only Fedora's system Python.

%install
install -d %{buildroot}%{kiki_python_sitelib}
cp -a src/kiki %{buildroot}%{kiki_python_sitelib}/

install -Dm0755 packaging/launcher/kiki %{buildroot}%{_bindir}/kiki

install -d %{buildroot}%{_datadir}/kiki
cp -a data/character %{buildroot}%{_datadir}/kiki/
cp -a data/icons %{buildroot}%{_datadir}/kiki/
install -Dm0644 data/io.github.projectkiki.Kiki.desktop \
  %{buildroot}%{_datadir}/kiki/io.github.projectkiki.Kiki.desktop

desktop-file-install --dir=%{buildroot}%{_datadir}/applications \
  data/io.github.projectkiki.Kiki.desktop
install -Dm0644 data/io.github.projectkiki.Kiki.metainfo.xml \
  %{buildroot}%{_metainfodir}/io.github.projectkiki.Kiki.metainfo.xml

for size in 64 128 256 512; do
  install -Dm0644 \
    data/icons/hicolor/${size}x${size}/apps/io.github.projectkiki.Kiki.png \
    %{buildroot}%{_datadir}/icons/hicolor/${size}x${size}/apps/io.github.projectkiki.Kiki.png
done

install -Dm0644 data/systemd/kiki.service \
  %{buildroot}%{_userunitdir}/kiki.service

install -Dm0755 scripts/setup-local-model.sh \
  %{buildroot}%{_libexecdir}/kiki/setup-local-model
install -Dm0755 scripts/setup-tts.sh \
  %{buildroot}%{_libexecdir}/kiki/setup-tts
# Globbed, not named: kiki_tts_server.py imports its siblings, and a module
# added here but forgotten in this list would ship a service that cannot start.
for part in services/qwen3-tts/*.py; do
  # streaming_spike.py is a measurement tool that pulls in torch; it is not
  # part of the service and must not reach an installed system.
  case "$(basename "$part")" in streaming_spike.py) continue ;; esac
  install -Dm0755 "$part" %{buildroot}%{_libexecdir}/kiki/"$(basename "$part")"
done
install -Dm0755 scripts/setup-llm.sh \
  %{buildroot}%{_libexecdir}/kiki/setup-llm
# Install every harness module rather than a hand-kept list: a file added to
# services/kiki-llm/ and forgotten here would ship a harness that cannot import
# itself.
for part in services/kiki-llm/*.py; do
  install -Dm0755 "${part}" %{buildroot}%{_libexecdir}/kiki/"$(basename "${part}")"
done
install -Dm0755 scripts/setup-stt.sh \
  %{buildroot}%{_libexecdir}/kiki/setup-stt
# Same glob rationale: kiki_stt_server.py is self-contained today, but a
# forgotten sibling would ship a service that cannot start.
for part in services/kiki-stt/*.py; do
  install -Dm0755 "${part}" %{buildroot}%{_libexecdir}/kiki/"$(basename "${part}")"
done
ln -s ../libexec/kiki/setup-local-model %{buildroot}%{_bindir}/kiki-setup-model
ln -s ../libexec/kiki/setup-tts %{buildroot}%{_bindir}/kiki-setup-tts
ln -s ../libexec/kiki/setup-llm %{buildroot}%{_bindir}/kiki-setup-llm
ln -s ../libexec/kiki/setup-stt %{buildroot}%{_bindir}/kiki-setup-stt

%check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src %{__python3} -m pytest -q
desktop-file-validate data/io.github.projectkiki.Kiki.desktop
appstreamcli validate --no-net --override=url-homepage-missing=info \
  data/io.github.projectkiki.Kiki.metainfo.xml
bash -n scripts/setup-local-model.sh scripts/setup-tts.sh scripts/setup-llm.sh \
  scripts/setup-stt.sh scripts/normalize-character-frame.sh

%post
%systemd_user_post kiki.service

%preun
%systemd_user_preun kiki.service

%postun
%systemd_user_postun_with_restart kiki.service

%files
%license LICENSE
%doc README.md docs/ARCHITECTURE.md docs/CHARACTER_DESIGN.md docs/GUIDE.md docs/DEVELOPER_GUIDE.md docs/VOICE_SUBSYSTEM.md docs/design/KIKI-v3-adult-concept.png
%{_bindir}/kiki
%{_bindir}/kiki-setup-model
%{_bindir}/kiki-setup-tts
%{_bindir}/kiki-setup-llm
%{_bindir}/kiki-setup-stt
%{_libexecdir}/kiki/
%{kiki_python_sitelib}/kiki/
%{_datadir}/kiki/
%{_datadir}/applications/io.github.projectkiki.Kiki.desktop
%{_metainfodir}/io.github.projectkiki.Kiki.metainfo.xml
%{_datadir}/icons/hicolor/*/apps/io.github.projectkiki.Kiki.png
%{_userunitdir}/kiki.service

%changelog
* Sat Aug 22 2026 Project KIKI <noreply@localhost> - 0.8.0-1
- Add KIKI's own LLM harness: PyTorch/transformers as kiki-llm.service, with
  token-level reasoning suppression, tool calling and a VRAM budget
- Add continuous batching: one forward pass serves every active sequence
  (measured 1.74x at two concurrent requests, 3.28x at four)
- Size the context window per turn instead of reserving a fixed one
- Report the active provider in --check instead of always Ollama

* Sat Aug 22 2026 Project KIKI <noreply@localhost> - 0.7.0-1
- Let KIKI open folders, files, terminal, editor and links herself at the new
  "trusted" level, confined to registered workspaces
- Add the LAUNCH risk class so opening is not classified as a data change
- Fix kiki-setup-tts, which could not find its server file when installed
- Fix the TTS voice: capitalised speaker names never matched the model's
  lower-case list, so KIKI always spoke with a male fallback voice
- Skip model deliberation by default (19.5 s to 5.2 s for a short reply)

* Sat Aug 22 2026 Project KIKI <noreply@localhost> - 0.6.0-1
- Let the model call read-only status tools itself, bounded and audited
- Add the local "KIKI" wake word, opt-in and off by default
- Add a visible local memory KIKI uses in later conversations
- Split the system prompt into fixed rules and a switchable persona
- Report low battery and full disks on KIKI's own initiative, with quiet hours

* Sat Aug 22 2026 Project KIKI <noreply@localhost> - 0.5.0-1
- Add controlled desktop actions and a Fedora 44 complete installer
- Make the default audio and desktop-control runtime dependencies explicit

* Sat Aug 22 2026 Project KIKI <noreply@localhost> - 0.4.0-1
- Enforce plan-session binding and bounded visible test output
- Harden session stop/workspace binding and make TTS synthesis cancellable

* Sat Aug 22 2026 Project KIKI <noreply@localhost> - 0.3.0-1
- Add the mature KIKI character pack with complete animated desktop states
- Make the adult design the default while retaining the original character pack

* Sat Aug 22 2026 Project KIKI <noreply@localhost> - 0.2.0-1
- Repair German-locale speech recognition and add local speech fallback
- Refine KIKI persona, model profiles, and desktop-pet animation timing

* Fri Aug 21 2026 Project KIKI <noreply@localhost> - 0.1.0-1
- First installable Fedora package
