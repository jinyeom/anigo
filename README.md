# &#127843; Anigo - Looped Image Generation with a Randomly Created CPPN
Anigo is an application that randomly generates a fixed-topology CPPN which
generates images in a sequence, so that they can form a loop of animated GIF.

## Installation
If you already have Go installed on your machine and want to build it yourself,
execute the following bash script. This script file is also provided in the repository.

```bash
go get github.com/gonum/matrix/mat64
go get github.com/kyokomi/emoji
go get github.com/whitewolf-studio/anigo
cd $GOPATH/src/github.com/whitewolf-studio/anigo
go install
```

## Example
```bash
anigo -name=example -width=200 -height=200 -seed=12345 -pattern -gray
```
![alt text](https://github.com/jinyeom/anigo/blob/master/screenshot.png "screenshot")

![alt text](https://github.com/jinyeom/anigo/blob/master/example.gif "example animation")

For help, execute on a terminal, `anigo -h`.
