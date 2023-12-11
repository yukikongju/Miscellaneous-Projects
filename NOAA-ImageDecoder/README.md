# NOAA Image Decoder

The [NOAA](https://noaa-apt.mbernardi.com.ar/index.html) satellite send a 
signal that can be decoded to an image of the earth. Our goal will be to 
capture that signal and to generate the image from that signal.

## Decoding using the NOAA-APT Software

**Installation**

The `noaa-apt` is a software to convert .wav file to .png files. The steps 
to download can be found [here](https://noaa-apt.mbernardi.com.ar/download.html)

**Usage from terminal**

```{sh}
> noaa-apt samples/noaa18_27072012_try1.wav -o output.png
```

## Retrieving your own .wav file from satellite and antenna


## Ressources

- [Medium - ](https://medium.com/swlh/decoding-noaa-satellite-images-using-50-lines-of-code-3c5d1d0a08da)
