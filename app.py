import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from photutils.detection import DAOStarFinder
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry
)

from scipy.ndimage import gaussian_filter

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Astropipe Professional",
    layout="wide"
)

st.title("🌌 Astropipe Professional FITS Photometry Pipeline")

st.markdown("""
This application performs:

- FITS image reduction
- Background estimation
- Source detection
- Aperture photometry
- Background subtraction
- Signal-to-noise calculation
- Instrumental magnitude estimation
- Optional WCS coordinate extraction
- CSV catalog export
""")

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("⚙️ Pipeline Controls")

threshold_factor = st.sidebar.slider(
    "Detection Threshold (Sigma)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)

fwhm_val = st.sidebar.slider(
    "FWHM (pixels)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5
)

aperture_radius = st.sidebar.slider(
    "Aperture Radius",
    min_value=2.0,
    max_value=15.0,
    value=6.0,
    step=0.5
)

annulus_inner = st.sidebar.slider(
    "Annulus Inner Radius",
    min_value=5.0,
    max_value=25.0,
    value=8.0,
    step=0.5
)

annulus_outer = st.sidebar.slider(
    "Annulus Outer Radius",
    min_value=8.0,
    max_value=35.0,
    value=12.0,
    step=0.5
)

zero_point = st.sidebar.number_input(
    "Photometric Zero Point",
    value=25.0
)

gaussian_sigma = st.sidebar.slider(
    "Gaussian Smoothing Sigma",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1
)

# Display Controls
st.sidebar.header("🖼️ Display Settings")

stretch_min = st.sidebar.slider(
    "Display Minimum Stretch",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

stretch_max = st.sidebar.slider(
    "Display Maximum Stretch",
    min_value=5.0,
    max_value=50.0,
    value=10.0,
    step=0.5
)

# Catalog Filter
st.sidebar.header("📊 Catalog Filtering")

mag_limit = st.sidebar.slider(
    "Maximum Magnitude",
    min_value=0.0,
    max_value=40.0,
    value=20.0,
    step=0.5
)

# =========================================================
# FILE UPLOADER
# =========================================================

import os

# 1. Add the toggle in the sidebar
input_method = st.sidebar.radio(
    "Select Data Source:", 
    ("Use Demo Sample", "Upload My Own")
)

uploaded_file = None

# 2. Logic to choose between the demo file or manual upload
if input_method == "Use Demo Sample":
    file_path = "sample_data/sample.fits"
    if os.path.exists(file_path):
        uploaded_file = file_path
    else:
        st.sidebar.error("Demo file not found!")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload FITS File",
        type=["fits", "fit", "fits.gz"]
    )


# =========================================================
# PROCESSING
# =========================================================

if uploaded_file is not None:

    with st.spinner("Processing astronomical image..."):

        try:
            # -------------------------------------------------
            # OPEN FITS
            # -------------------------------------------------

            hdul = fits.open(uploaded_file)

            data = hdul[0].data

            if data is None and len(hdul) > 1:
                data = hdul[1].data

            if data is None:
                st.error("No image data found in FITS file.")
                st.stop()

            # -------------------------------------------------
            # HANDLE MULTIDIMENSIONAL DATA
            # -------------------------------------------------

            while data.ndim > 2:
                data = data[0]

            data = np.asarray(data, dtype=float)

            # -------------------------------------------------
            # BACKGROUND STATS
            # -------------------------------------------------

            mean, median, std = sigma_clipped_stats(
                data,
                sigma=3.0
            )

            eff_std = std if std > 0 else 0.01

            # -------------------------------------------------
            # CLEAN BAD PIXELS
            # -------------------------------------------------

            data = np.nan_to_num(
                data,
                nan=median,
                posinf=median,
                neginf=median
            )

            # -------------------------------------------------
            # IMAGE SMOOTHING
            # -------------------------------------------------

            smoothed = gaussian_filter(
                data,
                sigma=gaussian_sigma
            )

            # -------------------------------------------------
            # SOURCE DETECTION
            # -------------------------------------------------

            daofind = DAOStarFinder(
                fwhm=fwhm_val,
                threshold=threshold_factor * eff_std
            )

            sources = daofind(smoothed - median)

            # -------------------------------------------------
            # METRICS DISPLAY
            # -------------------------------------------------

            st.subheader("📈 Reduction Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Background Median",
                f"{median:.2f}"
            )

            col2.metric(
                "Background Std Dev",
                f"{std:.2f}"
            )

            if sources is None or len(sources) == 0:
                col3.metric("Detected Sources", 0)
                st.warning("No sources detected.")
                st.stop()

            # -------------------------------------------------
            # EDGE FILTERING
            # -------------------------------------------------

            edge_margin = annulus_outer + 2

            mask = (
                (sources['xcentroid'] > edge_margin) &
                (sources['xcentroid'] < data.shape[1] - edge_margin) &
                (sources['ycentroid'] > edge_margin) &
                (sources['ycentroid'] < data.shape[0] - edge_margin)
            )

            sources = sources[mask]

            col3.metric(
                "Detected Sources",
                len(sources)
            )

            # -------------------------------------------------
            # APERTURE PHOTOMETRY
            # -------------------------------------------------

            positions = np.transpose((
                sources['xcentroid'],
                sources['ycentroid']
            ))

            apertures = CircularAperture(
                positions,
                r=aperture_radius
            )

            annulus = CircularAnnulus(
                positions,
                r_in=annulus_inner,
                r_out=annulus_outer
            )

            phot = aperture_photometry(
                data,
                [apertures, annulus]
            )

            # -------------------------------------------------
            # BACKGROUND SUBTRACTION
            # -------------------------------------------------

            bkg_mean = phot['aperture_sum_1'] / annulus.area
            bkg_sum = bkg_mean * apertures.area

            phot['background_flux'] = bkg_sum

            phot['flux_corrected'] = (
                phot['aperture_sum_0'] - bkg_sum
            )

            # -------------------------------------------------
            # SAFE FLUX
            # -------------------------------------------------

            safe_flux = np.clip(
                phot['flux_corrected'],
                1e-6,
                None
            )

            # -------------------------------------------------
            # MAGNITUDE
            # -------------------------------------------------

            phot['mag'] = (
                zero_point -
                2.5 * np.log10(safe_flux)
            )

            # -------------------------------------------------
            # SIGNAL TO NOISE
            # -------------------------------------------------

            phot['snr'] = (
                safe_flux /
                np.sqrt(np.abs(safe_flux) + bkg_sum)
            )

            # -------------------------------------------------
            # OPTIONAL WCS
            # -------------------------------------------------

            try:
                wcs = WCS(hdul[0].header)

                world = wcs.pixel_to_world(
                    sources['xcentroid'],
                    sources['ycentroid']
                )

                phot['ra_deg'] = world.ra.deg
                phot['dec_deg'] = world.dec.deg

                wcs_available = True

            except Exception:
                wcs_available = False

            # -------------------------------------------------
            # IMAGE DISPLAY
            # -------------------------------------------------

            st.subheader("🛰️ Source Detection Map")

            fig, ax = plt.subplots(figsize=(12, 10))

            vmin = max(
                median,
                1e-6
            )

            vmax = max(
                median + (stretch_max * eff_std),
                vmin + 1.0
            )

            image = ax.imshow(
                data,
                cmap='inferno',
                origin='lower',
                norm=LogNorm(
                    vmin=vmin,
                    vmax=vmax
                )
            )

            apertures.plot(
                color='cyan',
                lw=1.2,
                alpha=0.8
            )

            ax.set_title("Detected Sources")

            plt.colorbar(
                image,
                ax=ax,
                label="Flux"
            )

            st.pyplot(fig)

            # -------------------------------------------------
            # CONVERT TO DATAFRAME
            # -------------------------------------------------

            df = phot.to_pandas()

            # Rename coordinates
            if 'xcenter' in df.columns:
                df = df.rename(columns={'xcenter': 'x'})

            if 'ycenter' in df.columns:
                df = df.rename(columns={'ycenter': 'y'})

            # -------------------------------------------------
            # FILTER BY MAGNITUDE
            # -------------------------------------------------

            filtered_df = df[df['mag'] < mag_limit]

            # -------------------------------------------------
            # DISPLAY TABLE
            # -------------------------------------------------

            st.subheader("📋 Professional Photometry Catalog")

            display_columns = [
                c for c in [
                    'id',
                    'x',
                    'y',
                    'flux_corrected',
                    'background_flux',
                    'snr',
                    'mag',
                    'ra_deg',
                    'dec_deg'
                ]
                if c in filtered_df.columns
            ]

            st.dataframe(
                filtered_df[display_columns].head(100),
                use_container_width=True
            )

            # -------------------------------------------------
            # HISTOGRAM
            # -------------------------------------------------

            st.subheader("📉 Magnitude Distribution")

            fig2, ax2 = plt.subplots(figsize=(8, 5))

            ax2.hist(
                filtered_df['mag'],
                bins=30
            )

            ax2.set_xlabel("Magnitude")
            ax2.set_ylabel("Number of Sources")
            ax2.set_title("Magnitude Histogram")

            st.pyplot(fig2)

            # -------------------------------------------------
            # DOWNLOAD CSV
            # -------------------------------------------------

            csv = filtered_df.to_csv(
                index=False
            ).encode('utf-8')

            st.download_button(
                label="⬇️ Download Photometry Catalog",
                data=csv,
                file_name="astropipe_catalog.csv",
                mime="text/csv"
            )

            # -------------------------------------------------
            # WCS STATUS
            # -------------------------------------------------

            if wcs_available:
                st.success("WCS coordinates successfully extracted.")
            else:
                st.info("No valid WCS information found in FITS header.")

            # -------------------------------------------------
            # CLOSE FITS
            # -------------------------------------------------

            hdul.close()

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
