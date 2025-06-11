//frontendCalls3.js

// march 27 3 pm
document.addEventListener('DOMContentLoaded', () => {
  // --- Pre-populate path inputs from localStorage ---
  const rawDataPathInput = document.getElementById('rawDataPath');
  if (rawDataPathInput) {
    const storedRawPath = localStorage.getItem('rawDataPath');
    if (storedRawPath) {
      rawDataPathInput.value = storedRawPath;
    }
    rawDataPathInput.addEventListener('change', () => {
      localStorage.setItem('rawDataPath', rawDataPathInput.value.trim());
    });
  }

  const analyzedDataPathInput = document.getElementById('analyzedDataPath');
  if (analyzedDataPathInput) {
    const storedAnalyzedPath = localStorage.getItem('analyzedDataPath');
    if (storedAnalyzedPath) {
      analyzedDataPathInput.value = storedAnalyzedPath;
    }
    analyzedDataPathInput.addEventListener('change', () => {
      localStorage.setItem('analyzedDataPath', analyzedDataPathInput.value.trim());
    });
  }

  // Here we use the key "outputCCPath" for the output folder.
  const outputFolderInput = document.getElementById('inputOutputFolder');
  if (outputFolderInput) {
    const storedOutputFolder = localStorage.getItem('outputCCPath');
    if (storedOutputFolder) {
      outputFolderInput.value = storedOutputFolder;
    }
    outputFolderInput.addEventListener('change', () => {
      localStorage.setItem('outputCCPath', outputFolderInput.value.trim());
    });
  }

  // ------------------------------------------------------------------------
  // Generic helper to handle JSON from FastAPI or pass error forward
  // ------------------------------------------------------------------------
  async function handleApiResponse(response) {
    const contentType = response.headers.get("content-type");
    const clone = response.clone();
    let data;
    try {
      data = await response.json();
    } catch (err) {
      const txt = await clone.text();
      throw new Error(txt || 'Unknown error');
    }
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Unknown JSON error');
    }
    return data;
  }

  // ------------------------------------------------------------------------
  // 1) Placeholder for old skip_analysis logic
  // ------------------------------------------------------------------------
  const btnAnalyze = document.getElementById('btnAnalyze');
  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', async () => {
      alert("This is just a placeholder for your old skip_analysis logic.");
    });
  }

  // ------------------------------------------------------------------------
  // 2) Analyze Raw Data => POST /analyze-raw-data
  // ------------------------------------------------------------------------
  const btnAnalyzeRaw = document.getElementById('btnAnalyzeRaw');
  if (btnAnalyzeRaw) {
    btnAnalyzeRaw.addEventListener('click', async () => {
      try {
        const rawDataPath = document.getElementById('rawDataPath').value.trim();
        const digitPrefix = document.getElementById('digitPrefix').value.trim();
        const areaCodeStr = document.getElementById('inputAreaCode').value.trim();
        const forceRefresh = false;
  
        const payload = {
          raw_data_path: rawDataPath,
          digit_prefix: digitPrefix || null,
          area_selection: areaCodeStr ? parseInt(areaCodeStr) : null,
          output_cc_path: "myRawOutput",  // or read from a user input if desired
          force_refresh: forceRefresh
        };
  
        const resp = await fetch('/analyze-raw-data', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await handleApiResponse(resp);
        alert('AnalyzeRaw => ' + JSON.stringify(data));
      } catch (err) {
        alert(err.message);
      }
    });
  }

  // // ------------------------------------------------------------------------
  // // 3) Load & Filter Data => POST /load-and-filter-data
  // // ------------------------------------------------------------------------
  // const btnLoadFilter = document.getElementById('btnLoadFilter');
  // if (btnLoadFilter) {
  //   btnLoadFilter.addEventListener('click', async () => {
  //     try {
  //       const analyzedPath = document.getElementById('analyzedDataPath').value.trim();
  //       const areaSelection = document.getElementById('loadAreaSelection').value.trim();
  //       const digitPrefix = document.getElementById('loadDigitPrefix').value.trim();
  //       const outputCCPath = document.getElementById('inputOutputFolder').value.trim();
  //       const forceRefresh = false;
  
  //       const payload = {
  //         analyzed_data_path: analyzedPath,
  //         area_selection: areaSelection ? parseInt(areaSelection) : null,
  //         digit_prefix: digitPrefix || null,
  //         output_cc_path: outputCCPath,
  //         force_refresh: forceRefresh
  //       };
  //       const resp = await fetch('/load-and-filter-data', {
  //         method: 'POST',
  //         headers: { 'Content-Type': 'application/json' },
  //         body: JSON.stringify(payload)
  //       });
  //       const data = await handleApiResponse(resp);
  //       alert("Loaded & Filtered => " + JSON.stringify(data));
  
  //       // re-populate sample dropdowns
  //       populateSampleDropdowns();

  //       // ─── NEW: display the histogram if the backend gave us a URL ───
  //       if (data.histogram_url) {
  //       // find or create a container just below section 4
  //       let histContainer = document.getElementById('histogramContainer');
  //       if (!histContainer) {
  //         histContainer = document.createElement('div');
  //         histContainer.id = 'histogramContainer';
  //         // insert it right after your "4. Iterate Over Labels" div
  //         const section4 = document.getElementById('iterSection');

  //         // const section4 = document.querySelector('.iterSection');
  //         section4.insertAdjacentElement('afterend', histContainer);
  //       }
  //       // fill it with the heading + image
  //       histContainer.innerHTML = `
  //         <h3>Multiplicity Distribution of All MIDs</h3>
  //         <img src="${data.histogram_url}"
  //              alt="Multiplicity Histogram"
  //              style="max-width:100%; height:auto;" />
  //       `;
  //     }
  //     // ───────────────────────────────────────────────────────────────────────

  //     } catch (err) {
  //       alert(err.message);
  //     }
  //   });
  // }
  // ------------------------------------------------------------------------
  // 3) Load & Filter Data => POST /load-and-filter-data
  // ------------------------------------------------------------------------
  const btnLoadFilter = document.getElementById('btnLoadFilter');
  if (btnLoadFilter) {
    btnLoadFilter.addEventListener('click', async () => {
      try {
        const analyzedPath = document.getElementById('analyzedDataPath').value.trim();
        const areaSelection = document.getElementById('loadAreaSelection').value.trim();
        const digitPrefix = document.getElementById('loadDigitPrefix').value.trim();
        const outputCCPath = document.getElementById('inputOutputFolder').value.trim();
        const groupName = document.getElementById('loadGroupName').value.trim();
        const forceRefresh = false;

        const payload = {
          analyzed_data_path: analyzedPath,
          area_selection: areaSelection ? parseInt(areaSelection) : null,
          digit_prefix: digitPrefix || null,
          output_cc_path: outputCCPath,
          force_refresh: forceRefresh
        };
        // if (groupName) {
        //   // allow comma-separated list
        //   payload.groups = groupName
        //     .split(',')
        //     .map(g => g.trim())
        //     .filter(g => g.length > 0);
        // }
        payload.groups = groupName ? [groupName] : ["grp_0"];   // default grp_0

        const resp = await fetch('/load-and-filter-data', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await handleApiResponse(resp);
        alert("Loaded & Filtered => " + JSON.stringify(data));

        // Re-populate sample dropdowns
        populateSampleDropdowns();

        // ─── Display the Multiplicity Histogram ───
        if (data.histogram_url) {
          let histContainer = document.getElementById('histogramContainer');
          if (!histContainer) {
            histContainer = document.createElement('div');
            histContainer.id = 'histogramContainer';
            // Insert it right after your "4. Iterate Over Labels" section
            const section4 = document.getElementById('iterSection');
            section4.insertAdjacentElement('afterend', histContainer);
          }
          histContainer.innerHTML = `
            <h3>Multiplicity Distribution of All MIDs</h3>
            <img src="${data.histogram_url}"
                alt="Multiplicity Histogram"
                style="max-width:100%; height:auto;" />
          `;
        }

        // ─── Display the Energy Histogram (LID Energy Frequency) ───
        if (data.energy_histogram_url) {
          let energyHistContainer = document.getElementById('energyHistogramContainer');
          if (!energyHistContainer) {
            energyHistContainer = document.createElement('div');
            energyHistContainer.id = 'energyHistogramContainer';
            // Insert it right below the multiplicity histogram container if available,
            // otherwise insert it right after the "iterSection" element.
            const histContainer = document.getElementById('histogramContainer');
            if (histContainer) {
              histContainer.insertAdjacentElement('afterend', energyHistContainer);
            } else {
              const section4 = document.getElementById('iterSection');
              section4.insertAdjacentElement('afterend', energyHistContainer);
            }
          }
          energyHistContainer.innerHTML = `
            <h3>Energy Frequency Distribution (E ev)</h3>
            <img src="${data.energy_histogram_url}"
                alt="Energy Histogram"
                style="max-width:100%; height:auto;" />
          `;
        }

      } catch (err) {
        alert(err.message);
      }
    });
  }


  // ------------------------------------------------------------------------
  // 4) Populate sample <select> elements => GET /get-samples
  // ------------------------------------------------------------------------
  async function populateSampleDropdowns() {
    try {
      const resp = await fetch('/get-samples');
      const data = await handleApiResponse(resp);
      const samples = data.samples || [];
  
      const iterSampleEl = document.getElementById('iterSample');
      const showMatchedSampleEl = document.getElementById('showMatchedSample');
  
      if (iterSampleEl) {
        iterSampleEl.innerHTML = '<option value="" disabled selected>Select a sample</option>';
        samples.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s;
          opt.textContent = s;
          iterSampleEl.appendChild(opt);
        });
      }
  
      if (showMatchedSampleEl) {
        showMatchedSampleEl.innerHTML = '<option value="" disabled selected>Select a sample</option>';
        samples.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s;
          opt.textContent = s;
          showMatchedSampleEl.appendChild(opt);
        });
      }
    } catch (err) {
      console.error("populateSampleDropdowns error:", err);
    }
  }
  populateSampleDropdowns();

  // ------------------------------------------------------------------------
  // 5) Show Matched CC => POST /show-matched-cc
  // ------------------------------------------------------------------------
  const btnShowMatched = document.getElementById('btnShowMatched');
  const btnShowMatchedPrev = document.getElementById('btnShowMatchedPrev');
  const btnShowMatchedNext = document.getElementById('btnShowMatchedNext');
  const showMatchedSampleEl = document.getElementById('showMatchedSample');
  const showMatchedLabelIndexEl = document.getElementById('showMatchedLabelIndex');
  const showMatchedPlotsContainer = document.getElementById('showMatchedPlots-container');
  const showMatchedButtonsContainer = document.getElementById('showMatchedButtons-container');
  const showMatchedResult = document.getElementById('showMatchedResult');
  
  let showMatchedCurrentLabelIndex = parseInt(showMatchedLabelIndexEl ? showMatchedLabelIndexEl.value : "1") || 1;
  
  async function loadShowMatchedPlot() {
    if (showMatchedResult) showMatchedResult.textContent = 'Loading matched CC plot...';
    if (showMatchedPlotsContainer) showMatchedPlotsContainer.innerHTML = '';
    if (showMatchedButtonsContainer) showMatchedButtonsContainer.innerHTML = '';
  
    const sample_name = showMatchedSampleEl ? showMatchedSampleEl.value.trim() : '';
    const label_index = showMatchedCurrentLabelIndex;
  
    try {
      const payload = { start_sample: sample_name, target_label: label_index };
      const resp = await fetch('/show-matched-cc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await handleApiResponse(resp);
  
      const backend = data.fastApiResponse;
      if (!backend) {
        if (showMatchedResult) showMatchedResult.textContent = 'No fastApiResponse in the response.';
        return;
      }
  
      if (!backend.figure_path) {
        if (showMatchedResult) showMatchedResult.textContent = 'No figure_path in the response.';
        return;
      }
  
      // Create image element for the new plot
      const img = document.createElement('img');
      img.src = backend.figure_path;
      img.style.maxWidth = '600px';
      if (showMatchedPlotsContainer) showMatchedPlotsContainer.appendChild(img);
  
      if (showMatchedResult) {
        showMatchedResult.textContent = `Plot for label=${backend.current_label_index}`;
      }
  
      // Load clickable areas if they exist
      if (showMatchedButtonsContainer) showMatchedButtonsContainer.innerHTML = '';
      let clickableRespData = null;
      try {
        const filename = backend.figure_path.split('/').pop();
        const clickableUrl = `/get-clickable-areas?plot_filename=${encodeURIComponent(filename)}`;
        const clickableResp = await fetch(clickableUrl);
        const clickableJson = await handleApiResponse(clickableResp);
        clickableRespData = clickableJson.clickable_areas;
      } catch (err) {
        console.warn("No clickable areas or error loading them:", err);
      }
  
      if (clickableRespData && clickableRespData.clickableAreas) {
        clickableRespData.clickableAreas.forEach(area => {
          const btn = document.createElement('button');
          btn.textContent = `Select LID ${area.lid}`;
          btn.addEventListener('click', () => {
            const url = `/line-peaks-view?sample_name=${area.sample}&target_label=${area.label}&specific_lid=${area.lid}&use_modified_data=false`;
            window.open(url, '_blank');
          });
          if (showMatchedButtonsContainer) {
            showMatchedButtonsContainer.appendChild(btn);
          }
        });
      }
  
      if (btnShowMatchedPrev) {
        btnShowMatchedPrev.disabled = (backend.prev_label_index == null);
      }
      if (btnShowMatchedNext) {
        btnShowMatchedNext.disabled = (backend.next_label_index == null);
      }
  
    } catch (err) {
      if (showMatchedResult) {
        showMatchedResult.textContent = `Error: ${err.message}`;
      }
    }
  }
  
  if (btnShowMatched) {
    btnShowMatched.addEventListener('click', () => {
      showMatchedCurrentLabelIndex = parseInt(showMatchedLabelIndexEl ? showMatchedLabelIndexEl.value : "1") || 1;
      loadShowMatchedPlot();
    });
  }
  
  if (btnShowMatchedPrev) {
    btnShowMatchedPrev.addEventListener('click', async () => {
      try {
        const sample_name = showMatchedSampleEl ? showMatchedSampleEl.value.trim() : '';
        const current_label = showMatchedCurrentLabelIndex;
        const resp = await fetch('/show-matched-cc', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ start_sample: sample_name, target_label: current_label })
        });
        const data = await handleApiResponse(resp);
        const backend = data.fastApiResponse;
        if (backend && backend.prev_label_index != null) {
          showMatchedCurrentLabelIndex = backend.prev_label_index;
          if (showMatchedLabelIndexEl) {
            showMatchedLabelIndexEl.value = showMatchedCurrentLabelIndex;
          }
          loadShowMatchedPlot();
        }
      } catch (err) {
        console.error('Prev label error:', err);
        if (showMatchedResult) showMatchedResult.textContent = err.message;
      }
    });
  }
  
  if (btnShowMatchedNext) {
    btnShowMatchedNext.addEventListener('click', async () => {
      try {
        const sample_name = showMatchedSampleEl ? showMatchedSampleEl.value.trim() : '';
        const current_label = showMatchedCurrentLabelIndex;
        const resp = await fetch('/show-matched-cc', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ start_sample: sample_name, target_label: current_label })
        });
        const data = await handleApiResponse(resp);
        const backend = data.fastApiResponse;
        if (backend && backend.next_label_index != null) {
          showMatchedCurrentLabelIndex = backend.next_label_index;
          if (showMatchedLabelIndexEl) {
            showMatchedLabelIndexEl.value = showMatchedCurrentLabelIndex;
          }
          loadShowMatchedPlot();
        }
      } catch (err) {
        console.error('Next label error:', err);
        if (showMatchedResult) showMatchedResult.textContent = err.message;
      }
    });
  }
  
  // ------------------------------------------------------------------------
  // 6) Single-Label vs Multi-Label iteration => GET /plot-multi-sample-iter
  // ------------------------------------------------------------------------
  // We'll use a new tab approach so that the full page (with operational buttons) loads.
  // Note: To avoid duplicate element IDs, here we assume the iteration elements use the same IDs
  // as in section 4.
  const iterSample = document.getElementById('iterSample');
  const iterLabelIndex = document.getElementById('iterLabelIndex');
  const minMids = document.getElementById('min_mids_per_label');
  const maxSamples = document.getElementById('max_samples');
  const useModified = document.getElementById('use_modified_data');
  const btnIterPlot = document.getElementById('btnIterPlot');
  const iterResult = document.getElementById('iterResult');
  


  // const minMidCountEl = document.getElementById('min_mid_count');
  // const minMidCountStr = minMidCountEl ? minMidCountEl.value.trim() : '';
  // if (minMidCountStr !== '') {
  //     params.set('min_mid_count', minMidCountStr);
  // }

  // Suppose you have something like:
  const minMidCountEl = document.getElementById('min_mid_count');
  const minMidCountStr = minMidCountEl ? minMidCountEl.value.trim() : '';
  if (minMidCountStr !== '') {
    params.set('min_mid_count', minMidCountStr);
  }





  // if (btnIterPlot) {
  //   btnIterPlot.addEventListener('click', () => {
  //     if (iterResult) iterResult.textContent = 'Loading...';
  //     const sname = iterSample ? iterSample.value : '';
  //     if (!sname) {
  //       alert('No sample selected.');
  //       return;
  //     }
  
  //     const labelIndexStr = iterLabelIndex ? iterLabelIndex.value.trim() : '';
  //     const minMidsStr = minMids ? minMids.value.trim() : '';
  //     const maxSamplesVal = maxSamples ? maxSamples.value.trim() || '8' : '8';
  //     const useMod = useModified && useModified.checked ? 'true' : 'false';
  
  //     // NEW: Get desiredMid input value
  //     const desiredMidEl = document.getElementById('desiredMid');
  //     const desiredMidStr = desiredMidEl ? desiredMidEl.value.trim() : '';
  
  //     const params = new URLSearchParams();
  //     params.set('sample_name', sname);
  //     params.set('use_modified_data', useMod);
  //     params.set('max_samples', maxSamplesVal);
  //     if (labelIndexStr !== '') {
  //       params.set('label_index', labelIndexStr);
  //     }
  //     if (minMidsStr !== '') {
  //       params.set('min_mids_per_label', minMidsStr);
  //     }
  //     if (desiredMidStr !== '') {
  //       params.set('desired_mid', desiredMidStr);
  //     }
  //     const saveAllEl = document.getElementById('save_all_changes');
  //     params.set('save_changes', (saveAllEl && saveAllEl.checked) ? 'true' : 'false');
  
  //     // Build the URL with all query parameters and open it in a new tab
  //     const url = `/plot-multi-sample-iter?${params.toString()}`;
  //     window.open(url, "_blank");
  //   });
  // }
  
  if (btnIterPlot) {
    btnIterPlot.addEventListener('click', () => {
      if (iterResult) iterResult.textContent = 'Loading...';
      const sname = iterSample ? iterSample.value : '';
      if (!sname) {
        alert('No sample selected.');
        return;
      }
  
      const labelIndexStr = iterLabelIndex ? iterLabelIndex.value.trim() : '';
      const minMidsStr = minMids ? minMids.value.trim() : '';
      const maxSamplesVal = maxSamples ? maxSamples.value.trim() || '8' : '8';
      const useMod = useModified && useModified.checked ? 'true' : 'false';
  
      // NEW: Get desiredMid input value
      const desiredMidEl = document.getElementById('desiredMid');
      const desiredMidStr = desiredMidEl ? desiredMidEl.value.trim() : '';
  
      // NEW: Get min_mid_count input value for filtering labels
      const minMidCountEl = document.getElementById('min_mid_count');
      const minMidCountStr = minMidCountEl ? minMidCountEl.value.trim() : '';
  
      const params = new URLSearchParams();
      params.set('sample_name', sname);
      params.set('use_modified_data', useMod);
      params.set('max_samples', maxSamplesVal);
      if (labelIndexStr !== '') {
        params.set('label_index', labelIndexStr);
      }
      if (minMidsStr !== '') {
        params.set('min_mids_per_label', minMidsStr);
      }
      if (desiredMidStr !== '') {
        params.set('desired_mid', desiredMidStr);
      }
      if (minMidCountStr !== '') {
        params.set('min_mid_count', minMidCountStr);
      }
      const saveAllEl = document.getElementById('save_all_changes');
      params.set('save_changes', (saveAllEl && saveAllEl.checked) ? 'true' : 'false');
  
      const url = `/plot-multi-sample-iter?${params.toString()}`;
      window.open(url, "_blank");
    });
  }

  // ------------------------------------------------------------------------
  // 7) LID Editing Functions => Called inline, so attach them to window.*
  // ------------------------------------------------------------------------
  window.assignSubMid = function(sampleName, targetLabel, lid, inputId) {
    const val = document.getElementById(inputId).value;
    if (!val) {
      alert('Provide sub-mid');
      return;
    }
    const payload = {
      sample_name: sampleName,
      target_label: targetLabel,
      lid: lid,
      new_sub_mid: val
    };
    fetch('/assign-submid', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(d => {
      alert('Assigned Sub MID: ' + JSON.stringify(d));
      location.reload();
    })
    .catch(e => alert('Error:' + e));
  };
  
  window.undoSubMid = function(sampleName, targetLabel, lid) {
    const payload = {
      sample_name: sampleName,
      target_label: targetLabel,
      lid: lid
    };
    fetch('/undo-submid', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
      alert("Undo Successful: " + JSON.stringify(data));
      location.reload();
    })
    .catch(err => alert("Error: " + err.message));
  };
  
  window.renderLidControl = function(sampleName, targetLabel, lid, imageUrl) {
    const container = document.createElement('div');
    container.className = 'lid-control';
    container.innerHTML = `
      <img src="${imageUrl}" alt="LID ${lid}" style="max-width:150px;"><br>
      <input type="text" id="submid_${sampleName}_${targetLabel}_${lid}" placeholder="sub-mid?" style="width:50px;">
      <button onclick="assignSubMid('${sampleName}', ${targetLabel}, ${lid}, 'submid_${sampleName}_${targetLabel}_${lid}')">
        Assign
      </button>
      <button onclick="undoSubMid('${sampleName}', ${targetLabel}, ${lid})">
        Undo Sub MID
      </button>
    `;
    return container;
  };
  
  window.mergeLids = function(sampleName, targetLabel, currentLid) {
    let lidsInput = prompt("Enter the lids to merge (comma separated). If empty, current lid will be merged:", currentLid);
    if (!lidsInput) {
      lidsInput = "" + currentLid;
    }
    let lidsArray = lidsInput.split(',')
      .map(s => parseInt(s.trim()))
      .filter(n => !isNaN(n));
      
    let mergeComment = prompt("Enter merge comment (optional):", "");
    
    const payload = {
      sample_name: sampleName,
      target_label: targetLabel,
      lids: lidsArray,
      merge_comment: mergeComment
    };
    
    fetch('/merge-lids', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
      alert('Merge result: ' + JSON.stringify(data));
      location.reload();
    })
    .catch(e => alert('Error during merge: ' + e));
  };
  
  window.openLinePeaksView = function(sampleName, targetLabel, lid, useModified) {
    const url = `/line-peaks-view?sample_name=${sampleName}&target_label=${targetLabel}&specific_lid=${lid}&use_modified_data=${useModified}`;
    window.open(url, '_blank');
  };
  
  window.applyLabelChanges = async function(sampleName) {
    if (!confirm("Apply changes for all labels in sample " + sampleName + "?")) return;
    try {
      const payload = {
        sample_name: sampleName,
        comment: "Manual apply changes for all labels"
      };
      const resp = await fetch("/apply-label-changes", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await handleApiResponse(resp);
      alert(data.message);
      location.reload();
    } catch (err) {
      alert('Error applying label changes: ' + err.message);
    }
  };
  
  window.discardLabelChanges = async function(sampleName, labelVal) {
    if (!confirm("Discard changes for label=" + labelVal + "?")) return;
    try {
      const payload = {
        sample_name: sampleName,
        label: labelVal,
        target_label: labelVal
      };
      const resp = await fetch("/discard-label-changes", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await handleApiResponse(resp);
      alert(data.message);
      location.reload();
    } catch (err) {
      alert('Error discarding label changes: ' + err.message);
    }
  };

});
