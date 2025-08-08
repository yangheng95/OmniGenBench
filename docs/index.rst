.. .. OmniGenBench documentation master file, created by sphinx-quickstart
..    You can adapt this file completely to your liking, but it should at least
..    contain the root `toctree` directive.

.. Welcome to OmniGenBench Documentation !
.. ==========================

.. .. .. image:: ../asset/favicon.png
.. ..    :width: 1000px
.. ..    :align: left
.. ..    :alt: OmniGenBench Logo

.. .. raw:: html

..    <div style="text-align: center; margin: 2em 0; width: 1000px; align: left; display: flex; flex-direction: column;">
..    <p style="font-size: 1.2em; color: #666; max-width: 1000px; margin: 0 auto; line-height: 1.8;">
..    OmniGenBench offers an all-in-one solution for genomic foundation model finetuning, inference, deployment and automated benchmarking, designed for research and applications in genomics.
..    </p>
..    </div>

.. .. raw:: html

..    <div style="max-width: 1000px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 2em; border-radius: 12px; margin: 2em 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
..      <h3 style="margin-top: 0; color: #2c3e50;">✨ Key Features</h3>
..      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; margin-top: 1em;">

..        <div style="background: white; padding: 1.5em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
..          <h4 style="margin-top: 0; color: #3498db;">🧬 Multi-Modal Support</h4>
..          <p style="margin-bottom: 0; color: #666;">Available for both RNA and DNA modalities with comprehensive downstream tasks and foundation models, including fine-tuning and evaluation.</p>
..        </div>

..        <div style="background: white; padding: 1.5em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
..          <h4 style="margin-top: 0; color: #3498db;">⚡ Efficient Fine-tuning</h4>
..          <p style="margin-bottom: 0; color: #666;">Full LoRA integration for efficient foundation model fine-tuning with up to 90% reduced computational requirements. A 24GB Graphic Card is enough for all models.</p>
..        </div>

..        <div style="background: white; padding: 1.5em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
..          <h4 style="margin-top: 0; color: #3498db;">🔍 Interpretability</h4>
..          <p style="margin-bottom: 0; color: #666;">Diverse explanation methods for better model interpretability, including attention visualization and motif discovery.</p>
..        </div>

..        <div style="background: white; padding: 1.5em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
..          <h4 style="margin-top: 0; color: #3498db;">📊 Rich Benchmarks</h4>
..          <p style="margin-bottom: 0; color: #666;">As a foundation model benchmark tool, 5 curated benchmarks covering structure prediction, classification, and cross-species analysis.</p>
..        </div>


..      </div>
..    </div>

.. .. toctree::
..    :maxdepth: 2
..    :caption: Installation

..    installation

.. .. toctree::
..    :maxdepth: 2
..    :caption: Basic Usage

..    usage

.. .. toctree::
..    :maxdepth: 2
..    :caption: Command Usage

..    cli

.. .. toctree::
..    :maxdepth: 2
..    :caption: Package Design Principles

..    design_principle

.. .. toctree::
..    :maxdepth: 2
..    :caption: API Reference

..    api_reference

.. For more details, refer to the navigation bar on the left.



.. .. image:: ../asset/favicon.png
..    :width: 150px
..    :align: center
..    :alt: OmniGenBench Logo

.. .. raw:: html

..    <div style="text-align: center; margin-top: 1em;">
..       <h1 style="font-size: 2.5em; font-weight: bold;">OmniGenBench</h1>
..    </div>

..    <div style="text-align: center; margin: 1em 0; width: 100%; display: flex; flex-direction: column; align-items: center;">
..       <p style="font-size: 1.25em; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
..          An all-in-one solution for genomic foundation model finetuning, inference, deployment, and automated benchmarking.
..       </p>
..    </div>

..    <div style="text-align: center; margin-top: 2em; margin-bottom: 2em; display: flex; justify-content: center; gap: 0.5em;">
..       <a href="installation.html" class="btn btn-primary" style="font-size: 1em; padding: 0.7em 1.2em;">Get Started</a>
..       <a href="https://github.com/your-repo/OmniGenBench" class="btn btn-secondary" style="font-size: 1em; padding: 0.7em 1.2em;">View on GitHub</a>
..    </div>

.. .. raw:: html

..    <div style="max-width: 1000px; margin: 3em auto;">
..      <h2 style="text-align: center; font-weight: 600; margin-bottom: 1.5em; font-size: 1.8em;">Key Features</h2>
..      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5em;">

..        <div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #e9ecef;">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🧬 Multi-Modal Support</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Available for both <strong>RNA and DNA</strong> modalities with comprehensive downstream tasks and foundation models, including fine-tuning and evaluation.</p>
..        </div>

..        <div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #e9ecef;">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">⚡ Efficient Fine-tuning</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Full <strong>LoRA integration</strong> for efficient foundation model fine-tuning with up to 90% reduced computational requirements. A 24GB GPU is enough for all models.</p>
..        </div>

..        <div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #e9ecef;">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🔍 Interpretability</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Diverse explanation methods for better model interpretability, including <strong>attention visualization</strong> and <strong>motif discovery</strong>.</p>
..        </div>

..        <div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #e9ecef;">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📊 Rich Benchmarks</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Includes <strong>5 curated benchmarks</strong> covering structure prediction, classification, and cross-species analysis to serve as a robust evaluation tool.</p>
..        </div>

..      </div>
..    </div>


.. .. toctree::
..    :maxdepth: 2
..    :caption: GETTING STARTED
..    :hidden:

..    installation
..    usage

.. .. toctree::
..    :maxdepth: 2
..    :caption: GUIDES
..    :hidden:

..    cli
..    design_principle

.. .. toctree::
..    :maxdepth: 2
..    :caption: API REFERENCE
..    :hidden:

..    api_reference










.. .. raw:: html

..    <style>
..      @keyframes gradient-animation {
..        0% {
..          background-position: 0% 50%;
..        }
..        50% {
..          background-position: 100% 50%;
..        }
..        100% {
..          background-position: 0% 50%;
..        }
..      }

..      body {
..        /* The animated gradient background */
..        background: linear-gradient(-45deg, #e7f0ff, #f5f7fa, #e8f5e9, #fff3e0);
..        background-size: 400% 400%;
..        animation: gradient-animation 25s ease infinite;
..      }
..    </style>

.. .. raw:: html

..    <div style="padding: 2.5em; margin: 3em auto; max-width: 1000px; background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 12px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); border: 1px solid rgba(255, 255, 255, 0.18);">
..       <div style="text-align: center;">
..          <h1 style="font-size: 2.8em; font-weight: bold; margin: 0;">OmniGenBench</h1>
..       </div>

..       <div style="text-align: center; margin: 1.5em 0; width: 100%; display: flex; flex-direction: column; align-items: center;">
..          <p style="font-size: 1.25em; color: #333; max-width: 800px; margin: 0 auto; line-height: 1.6;">
..             An all-in-one solution for genomic foundation model finetuning, inference, deployment, and automated benchmarking.
..          </p>
..       </div>

..       <div style="text-align: center; margin-top: 2em; margin-bottom: 1em; display: flex; justify-content: center; gap: 0.8em;">
..          <a href="installation.html" class="btn btn-primary" style="font-size: 1em; padding: 0.7em 1.5em; border-radius: 8px;">Get Started</a>
..          <a href="https://github.com/your-repo/OmniGenBench" class="btn btn-secondary" style="font-size: 1em; padding: 0.7em 1.5em; border-radius: 8px;">View on GitHub</a>
..       </div>
..    </div>

.. .. raw:: html

..    <div style="max-width: 1000px; margin: 3em auto;">
..      <h2 style="text-align: center; font-weight: 600; margin-bottom: 1.5em; font-size: 1.8em; color: #2c3e50;">Key Features</h2>
..      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5em;">

..        <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🧬 Multi-Modal Support</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Available for both <strong>RNA and DNA</strong> modalities with comprehensive downstream tasks and foundation models, including fine-tuning and evaluation.</p>
..        </div>

..        <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">⚡ Efficient Fine-tuning</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Full <strong>LoRA integration</strong> for efficient foundation model fine-tuning with up to 90% reduced computational requirements. A 24GB GPU is enough for all models.</p>
..        </div>

..        <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🔍 Interpretability</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Diverse explanation methods for better model interpretability, including <strong>attention visualization</strong> and <strong>motif discovery</strong>.</p>
..        </div>

..        <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📊 Rich Benchmarks</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Includes <strong>5 curated benchmarks</strong> covering structure prediction, classification, and cross-species analysis to serve as a robust evaluation tool.</p>
..        </div>

..      </div>
..    </div>

.. .. .. toctree::
.. ..    :maxdepth: 2
.. ..    :caption: Installation

.. ..    installation

.. .. .. toctree::
.. ..    :maxdepth: 2
.. ..    :caption: Basic Usage

.. ..    usage

.. .. .. toctree::
.. ..    :maxdepth: 2
.. ..    :caption: Command Usage

.. ..    cli

.. .. .. toctree::
.. ..    :maxdepth: 2
.. ..    :caption: Package Design Principles

.. ..    design_principle

.. .. .. toctree::
.. ..    :maxdepth: 2
.. ..    :caption: API Reference

.. ..    api_reference

.. .. For more details, refer to the navigation bar on the left.


.. .. raw:: html

..    <style>
..      .nav-card {
..        display: block;
..        text-decoration: none;
..        color: inherit;
..        background: rgba(255, 255, 255, 0.85);
..        padding: 1.5em;
..        border-radius: 12px;
..        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
..        border: 1px solid rgba(255, 255, 255, 0.2);
..        transition: transform 0.2s ease, box-shadow 0.2s ease;
..      }
..      .nav-card:hover {
..        transform: translateY(-5px);
..        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
..      }
..    </style>

..    <div style="max-width: 1000px; margin: 3em auto;">
..      <h2 style="text-align: center; font-weight: 600; margin-bottom: 1.5em; font-size: 1.8em; color: #2c3e50;">Explore the Documentation</h2>
..      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5em;">

..        <!-- Card 1: Installation -->
..        <a href="installation.html" class="nav-card">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🚀 Installation</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Start your journey by setting up OmniGenBench on your system.</p>
..        </a>

..        <!-- Card 2: Basic Usage -->
..        <a href="usage.html" class="nav-card">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📖 Basic Usage</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Learn the fundamental workflows and how to run your first task.</p>
..        </a>

..        <!-- Card 3: Command Usage -->
..        <a href="cli.html" class="nav-card">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🛠️ Command Usage (CLI)</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Master the command-line interface for powerful and flexible operations.</p>
..        </a>

..        <!-- Card 4: Design Principles -->
..        <a href="design_principle.html" class="nav-card">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🏗️ Design Principles</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Understand the core architecture and design choices behind the library.</p>
..        </a>

..        <!-- Card 5: API Reference -->
..        <a href="api_reference.html" class="nav-card">
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📚 API Reference</h4>
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Get detailed information about all public classes, functions, and methods.</p>
..        </a>

..      </div>
..    </div>

.. ..
..    ################################################################################
..    # The hidden toctree below is crucial.
..    # It is not displayed on the page, but it tells Sphinx how to build the
..    # navigation sidebar on the left.
..    ################################################################################

.. .. toctree::
..    :hidden:
..    :maxdepth: 2
..    :caption: Documentation

..    installation
..    usage
..    cli
..    design_principle
..    api_reference






.. .. raw:: html

..    <style>
..      /* --- 1. 动态背景动画 --- */
..      @keyframes gradient-animation {
..        0% { background-position: 0% 50%; }
..        50% { background-position: 100% 50%; }
..        100% { background-position: 0% 50%; }
..      }

..      body {
..        background: linear-gradient(-45deg, #e7f0ff, #f5f7fa, #e8f5e9, #fff3e0);
..        background-size: 400% 400%;
..        animation: gradient-animation 25s ease infinite;
..      }

..      /* --- 2. 强制居中页面主标题 --- */
..      .page .article-container h2 {
..          text-align: center;
..      }

..      /* --- 3. 导航卡片样式 --- */
..      .nav-card {
..        display: block; text-decoration: none; color: inherit;
..        background: rgba(255, 255, 255, 0.85);
..        padding: 1.5em; border-radius: 12px;
..        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
..        border: 1px solid rgba(255, 255, 255, 0.2);
..        transition: transform 0.2s ease, box-shadow 0.2s ease;
..      }
..      .nav-card:hover {
..        transform: translateY(-5px);
..        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
..      }
..    </style>

.. .. raw:: html

..    <div style="padding: 2.5em; margin-top: 1em; margin-bottom: 3em; background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 12px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); border: 1px solid rgba(255, 255, 255, 0.18);">
..       <div style="text-align: center;">
..          <img src="../asset/favicon.png" alt="OmniGenBench Logo" style="width: 100px; margin-bottom: 1em;"/>
..          <h1 style="font-size: 2.8em; font-weight: bold; margin: 0;">OmniGenBench</h1>
..       </div>
..       <div style="text-align: center; margin: 1.5em 0; width: 100%; display: flex; flex-direction: column; align-items: center;">
..          <p style="font-size: 1.25em; color: #333; max-width: 800px; margin: 0 auto; line-height: 1.6;">An all-in-one solution for genomic foundation model finetuning, inference, deployment, and automated benchmarking.</p>
..       </div>
..       <div style="text-align: center; margin-top: 2em; margin-bottom: 1em; display: flex; justify-content: center; gap: 0.8em;">
..          <a href="installation.html" class="btn btn-primary" style="font-size: 1em; padding: 0.7em 1.5em; border-radius: 8px;">Get Started</a>
..          <a href="https://github.com/your-repo/OmniGenBench" class="btn btn-secondary" style="font-size: 1em; padding: 0.7em 1.5em; border-radius: 8px;">View on GitHub</a>
..       </div>
..    </div>

.. Key Features
.. ------------

.. .. raw:: html

..    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5em;">
..        <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🧬 Multi-Modal Support</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Available for both <strong>RNA and DNA</strong> modalities with comprehensive downstream tasks and foundation models, including fine-tuning and evaluation.</p> </div>
..        <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">⚡ Efficient Fine-tuning</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Full <strong>LoRA integration</strong> for efficient foundation model fine-tuning with up to 90% reduced computational requirements. A 24GB GPU is enough for all models.</p> </div>
..        <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🔍 Interpretability</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Diverse explanation methods for better model interpretability, including <strong>attention visualization</strong> and <strong>motif discovery</strong>.</p> </div>
..        <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📊 Rich Benchmarks</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Includes <strong>5 curated benchmarks</strong> covering structure prediction, classification, and cross-species analysis to serve as a robust evaluation tool.</p> </div>
..    </div>

.. Explore the Documentation
.. -------------------------

.. .. raw:: html

..    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5em; margin-top: 1em;">
..        <a href="installation.html" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🚀 Installation</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Start your journey by setting up OmniGenBench on your system.</p> </a>
..        <a href="usage.html" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📖 Basic Usage</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Learn the fundamental workflows and how to run your first task.</p> </a>
..        <a href="cli.html" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🛠️ Command Usage (CLI)</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Master the command-line interface for powerful and flexible operations.</p> </a>
..        <a href="design_principle.html" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🏗️ Design Principles</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Understand the core architecture and design choices behind the library.</p> </a>
..        <a href="api_reference.html" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📚 API Reference</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Get detailed information about all public classes, functions, and methods.</p> </a>
..    </div>

.. .. toctree::
..    :hidden:
..    :maxdepth: 2
..    :caption: NAVIGATION

..    installation
..    usage
..    cli
..    design_principle
..    api_reference











.. .. raw:: html

..    <style>
..      /* --- 1. 动态背景动画 --- 
..      @keyframes gradient-animation {
..        0% { background-position: 0% 50%; }
..        50% { background-position: 100% 50%; }
..        100% { background-position: 0% 50%; }
..      }*/

..      /*body {
..        background: linear-gradient(-45deg, #e7f0ff, #f5f7fa, #e8f5e9, #fff3e0);
..        background-size: 400% 400%;
..        animation: gradient-animation 25s ease infinite;
..      }*/
     

..      /* --- 2. [新] 强制居中页面主标题 --- */
..      /* 我们通过选择 Furo 主题生成的主要内容容器内的 h2 元素来实现 */
..      .page .article-container h2 {
..          text-align: center;
..      }

..      /* --- 3. [新] 导航卡片样式 --- */
..      .nav-card {
..        display: block; text-decoration: none; color: inherit;
..        background: rgba(255, 255, 255, 0.85);
..        padding: 1.5em; border-radius: 12px;
..        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
..        border: 1px solid rgba(255, 255, 255, 0.2);
..        transition: transform 0.2s ease, box-shadow 0.2s ease;
..      }
..      .nav-card:hover {
..        transform: translateY(-5px);
..        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
..      }
..    </style>

.. .. raw:: html

..    <div style="padding: 2.5em; margin-top: 1em; margin-bottom: 3em; background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 12px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); border: 1px solid rgba(255, 255, 255, 0.18);">
..       <div style="text-align: center;">
..          <h1 style="font-size: 2.8em; font-weight: bold; margin: 0;">OmniGenBench</h1>
..       </div>
..       <div style="text-align: center; margin: 1.5em 0; width: 100%; display: flex; flex-direction: column; align-items: center;">
..          <p style="font-size: 1.25em; color: #333; max-width: 800px; margin: 0 auto; line-height: 1.6;">An all-in-one solution for genomic foundation model finetuning, inference, deployment, and automated benchmarking.</p>
..       </div>
..       <div style="text-align: center; margin-top: 2em; margin-bottom: 1em; display: flex; justify-content: center; gap: 0.8em;">
..          <a href="installation.rst" class="btn btn-primary" style="font-size: 1em; padding: 0.7em 1.5em; border-radius: 8px;">Get Started</a>
..          <a href="https://github.com/your-repo/OmniGenBench" class="btn btn-secondary" style="font-size: 1em; padding: 0.7em 1.5em; border-radius: 8px;">View on GitHub</a>
..       </div>
..    </div>


.. Key Features
.. ------------

.. .. raw:: html

..    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5em;">
..       <!-- 卡片1: Multi-Modal Support -->
..       <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);"> 
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🧬 Multi-Modal Support</h4> 
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Available for both <strong>RNA and DNA</strong> modalities with comprehensive downstream tasks and foundation models, including fine-tuning and evaluation.</p> 
..       </div>
..       <!-- 卡片2: Efficient Fine-tuning -->
..       <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);"> 
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">⚡ Efficient Fine-tuning</h4> 
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Full <strong>LoRA integration</strong> for efficient foundation model fine-tuning with up to 90% reduced computational requirements. A 24GB GPU is enough for all models.</p> 
..       </div>
..       <!-- 卡片3: Interpretability -->
..       <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);"> 
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🔍 Interpretability</h4> 
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Diverse explanation methods for better model interpretability, including <strong>attention visualization</strong> and <strong>motif discovery</strong>.</p> 
..       </div>
..       <!-- 卡片4: Rich Benchmarks -->
..       <div style="background: rgba(255, 255, 255, 0.85); padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(255, 255, 255, 0.2);"> 
..          <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📊 Rich Benchmarks</h4> 
..          <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Includes <strong>5 curated benchmarks</strong> covering structure prediction, classification, and cross-species analysis to serve as a robust evaluation tool.</p> 
..       </div>
..    </div>

.. Explore the Documentation
.. -------------------------

############
OmniGenBench
############


OmniGenBench is an **All-in-One** solution for genomic foundation model finetuning, inference, deployment, and automated benchmarking.


.. .. raw:: html

..    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5em; margin-top: 1em;">
..        <a href="installation.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🚀 Installation</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Start your journey by setting up OmniGenBench on your system.</p> </a>
..        <a href="usage.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📖 Basic Usage</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Learn the fundamental workflows and how to run your first task.</p> </a>
..        <a href="cli.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🛠️ Command Usage (CLI)</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Master the command-line interface for powerful and flexible operations.</p> </a>
..        <a href="design_principle.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🏗️ Design Principles</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Understand the core architecture and design choices behind the library.</p> </a>
..        <a href="api_reference.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📚 API Reference</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Get detailed information about all public classes, functions, and methods.</p> </a>
..    </div>


.. raw:: html

   <div class="nav-grid">
       <a href="installation.html" class="nav-card">
           <h4 class="nav-card-title">🚀 Installation</h4>
           <p class="nav-card-description">Start your journey by setting up OmniGenBench on your system.</p>
       </a>
       <a href="usage.html" class="nav-card">
           <h4 class="nav-card-title">📖 Basic Usage</h4>
           <p class="nav-card-description">Learn the fundamental workflows and how to run your first task.</p>
       </a>
       <a href="cli.html" class="nav-card">
           <h4 class="nav-card-title">🛠️ Command Usage (CLI)</h4>
           <p class="nav-card-description">Master the command-line interface for powerful and flexible operations.</p>
       </a>
       <a href="design_principle.html" class="nav-card">
           <h4 class="nav-card-title">🏗️ Design Principles</h4>
           <p class="nav-card-description">Understand the core architecture and design choices behind the library.</p>
       </a>
       <a href="api_reference.html" class="nav-card">
           <h4 class="nav-card-title">📚 API Reference</h4>
           <p class="nav-card-description">Get detailed information about all public classes, functions, and methods.</p>
       </a>
   </div>



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   installation

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Core Usage Guide

   usage

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Command Usage Examples

   cli

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Package Design Principles

   design_principle

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   api_reference



