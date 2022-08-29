/*
* StringReplacer.c
*
*  Created on: Sep 24, 2017
*      Author: HeMe2
*/

#include "stringreplacer.h"

namespace luojianet_ms {

  string StringReplacer::replaceSubstrings(const string &origin, const string & toReplace, const string & replacement) {
    size_t position = StringReplacer::searchForSubstring(origin, toReplace);

    if (position == origin.size()) {
      return origin;
    }
    else {
      const string &remainingString = origin.substr(position + toReplace.size(), origin.size());
      return origin.substr(0, position) +
        replacement +
        StringReplacer::replaceSubstrings(remainingString, toReplace, replacement);
    }
  }

  size_t StringReplacer::searchForSubstring(const string &origin, const string & toSearch) {
    // boundary checks first
    size_t bytesToCompare = toSearch.size();
    if (bytesToCompare < origin.size()) {
      // search from front to back
      for (size_t index = 0; index < origin.size(); index++) {
        if (StringReplacer::matches(origin.substr(index, bytesToCompare).c_str(), toSearch.c_str(), bytesToCompare)) {
          return index;
        }
      }
    }
    return origin.size();
  }

  bool StringReplacer::matches(const char * one, const char * two, size_t bytes) {
    if (bytes > 0) {
      if (*one == *two) {
        return matches(one + 1, two + 1, bytes - 1);
      }
      else {
        return false;
      }
    }
    return true;
  }

}
