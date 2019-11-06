/*
 * Copyright (c) 2009-2012 jMonkeyEngine
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of 'jMonkeyEngine' nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package edu.stanford.graphics.shapenet.jme3.asset;

import com.jme3.asset.*;

import java.io.*;

/**
 * <code>FileLocator</code> allows you to specify a folder where to
 * look for assets. 
 * @author Angel Chang
 */
public class MyFileLocator implements AssetLocator {

  private File root;

  public void setRootPath(String rootPath) {
    if (rootPath == null) {
      throw new IllegalArgumentException("rootPath is required: use '\' to register any file");
    } else if (rootPath == "/") {
      // any file is okay
      root = null;
      return;
    }

    try {
      root = new File(rootPath).getCanonicalFile();
      if (!root.isDirectory()){
        throw new IllegalArgumentException("Given root path \"" + root + "\" is not a directory");
      }
    } catch (IOException ex) {
      throw new AssetLoadException("Root path is invalid", ex);
    }
  }

  public AssetInfo locate(AssetManager manager, AssetKey key) {
    String name = key.getName();
    File file = (root != null)? new File(root, name) : new File(name);
    if (file.exists() && file.isFile()){
      try {
        // Now, check asset name requirements
        String canonical = file.getCanonicalPath();
        String absolute = file.getAbsolutePath();
        if (!canonical.endsWith(absolute)){
          throw new AssetNotFoundException("Asset name doesn't match requirements.\n"+
              "\"" + canonical + "\" doesn't match \"" + absolute + "\"");
        }
      } catch (IOException ex) {
        throw new AssetLoadException("Failed to get file canonical path " + file, ex);
      }

      AssetInfo fileAssetInfo = new AssetInfoFile(manager, key, file);
      return UncompressedAssetInfo.create(fileAssetInfo);
    }else{
      return null;
    }
  }

}
