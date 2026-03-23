# Developer Software: FAST

Getting software at Boeing can be an exercise in frustration. Devops-Setup minimizes this frustration and makes getting common developer software much easier and faster.

Devops-Setup is a simple 'application' (collection of scripts) used to automate the installation of various software. Its main use case is getting Boeing approved standard and non-standard software installed or updated quickly. Nearly all of the software that you can download using Devops-Setup is sourced from Artifactory (SRES). If you'd like something added, please feel free to create an issue, contact the developers/community on Mattermost, or contribute the changes yourself.

## Capabilities of Devops-Setup

- Get software fast and install it automatically
- Automatically set up and configure popular tools like Gradle, Maven, NPM, NuGet (and more) for use at Boeing
- Create Lists/Kits specifically for your team or group which can be used to efficiently set up new Developers and/or machines for your specific work
- Use a GUI to select software you want à la carte, or use kits to automatically install a predefined set of software

## Windows

## Kits

There are also predefined **kit-lists** which include all software packages for a particular use-case or set of work. There are several defined in this project, but you can write and add your own with a merge request.

You can find existing kit-lists [here](link-to-kit-lists).

## Single Software Packages

To install a single software package, locate its directory under `components`. Located in the directory should be an `Install.ps1` PowerShell script and a `README.md` containing the command to run it, as well as any specific concerns for the package.

## Package Versioning

Some software packages allow you to specify the version to install. The version is specified by setting a variable, before running the main DevOps-Setup script. The variables are generally named `DS_{software}_VERSION`. For example, to install IntelliJ IDEA v2025.1.1.1:

```powershell
$DS_INTELLIJ_VERSION = "2025.1.1.1"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::TLs12
```

For more information on software package versioning, see `CONTRIBUTING.md` and individual package readme files.

## Debugging

Devops-Setup supports verbose logging you can enable to test new features or debug problems. Enabling it is simple. Start a PowerShell prompt before running the Devops-Setup command and set the variable `$DebugPreference="Continue"`:

```powershell
$DebugPreference="Continue" # Enables Debugging

# Start Devops-Setup
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::TLs12
Invoke-Expression ((new-object net.webclient).DownloadString('https://git.web.boeing.com/DevHub/devops-setup/-/raw/master/Invoke-DevOpsSetup.ps1'))
```

Disable debugging by setting `$DebugPreference` to its default value of `"SilentlyContinue"`:

```powershell
$DebugPreference="SilentlyContinue" # Disables Debugging

# Starts Devops-Setup
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::TLs12
Invoke-Expression ((new-object net.webclient).DownloadString('https://git.web.boeing.com/DevHub/devops-setup/-/raw/master/Invoke-DevOpsSetup.ps1'))
```

## Contributing

If you have a change you would like to have merged into the project, fork this project into your personal project space, make the edits/changes, commit and push them to your personal forked project, then create a merge request. Refer to `CONTRIBUTING.md` for details.

## Testing

### Development Forks/Branches

Devops-Setup supports running from forks of the primary Devops-Setup GitLab repo or from other branches on the main repo. This means we can test changes properly before merging. Previously, testing was not reliable due to hardcoded values and had to be done by running individual scripts, making it very inconvenient and error prone.

To test from a fork, set the `$DS_BASE_URL` variable to the URL of the desired fork repo's root, prior to running Devops-Setup (or any of its functions). If `$DS_BASE_URL` is not set, the default is `https://git.web.boeing.com/DevHub/devops-setup`.

**Note:** forked project visibility must be set to "public".

To test a specific branch, set the `$DS_TARGET_REF` variable to the desired branch name. If `$DS_TARGET_REF` is not set, the default is `master`.

If both variables are set, the specified branch within the specified fork will be used:

```powershell
# Use fork 'my-fork'
$DS_BASE_URL="https://git.web.boeing.com/my-fork/devops-setup"

# Use branch 'my-branch'
$DS_TARGET_REF="my-branch"

# Starts Devops-Setup
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::TLs12
Invoke-Expression ((new-object net.webclient).DownloadString("$DS_BASE_URL/-/raw/$DS_TARGET_REF/Invoke-DevOpsSetup.ps1"))
```

### Individual Component Installer

You can test an individual component installer by:

1. Cloning this repo
2. Opening in Visual Studio Code
3. Running `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` in attached PowerShell terminal
4. Opening desired component installer
5. Selecting appropriate option from Run menu

## Linux

No version of Devops-Setup exists anymore for Linux. Previous implementation for Linux can be found at [I617](link-to-I617) (merged).

## Support

Devops-Setup is a community supported tool run on a best effort basis.

For support, please ask on our [Devops-Setup Mattermost channel](link-to-mattermost).

For bugs and feature/installer requests, please submit an issue or create a merge request. See our [Contributing Guide](CONTRIBUTING.md).
